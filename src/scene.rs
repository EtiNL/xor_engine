use std::error::Error;
use std::f32::consts::PI;

use crate::ecs::ecs::{World, Entity};
use crate::ecs::ecs::components::{
    Transform, MaterialComponent, SdfBase, SdfType, Rotating, SpaceFolding, Axis,
    TextureManager,
    CsgTree, Node, NodeType, OperationType, NodeKey
};
use crate::ecs::ecs::{Vec3, Quat, Mat3};

/// Spawn an orthonormal-basis gizmo (X/Y/Z):
/// each axis = one Line (shaft) + one Cone (arrowhead).
/// Returns the *root* entity so you can move/rotate it.
pub fn spawn_basis_gizmo(
    world: &mut World,
    origin: Vec3,
    axis_len: f32,
    shaft_radius: f32,
    cone_h: f32,
    cone_base_r: f32,
) -> Result<Entity, Box<dyn std::error::Error>> {
    let root = world.spawn_group("basis_gizmo");
    world.insert_transform(root, Transform { position: origin, rotation: Quat::identity() });

    let col_x = [0.90, 0.25, 0.25];
    let col_y = [0.25, 0.90, 0.25];
    let col_z = [0.25, 0.45, 0.95];

    // Map local +Z to axis
    let rot_w_to_x = Quat::from_axis_angle(Vec3::Y, (90.0f32).to_radians()); // Z→X
    let rot_w_to_y = Quat::from_axis_angle(Vec3::X, (-90.0f32).to_radians()); // Z→Y
    let rot_w_to_z = Quat::identity();                                        // Z→Z

    // Build one axis entirely in *local* space of the gizmo root
    let mut build_axis = |axis_dir: Vec3, rot: Quat, color: [f32;3]| -> Result<(), Box<dyn std::error::Error>> {
        // Shaft
        let e_line = world.spawn();
        world.insert_sdf_base(e_line, SdfBase { sdf_type: SdfType::Line, params: [axis_len, shaft_radius, 0.0] });
        world.insert_material(e_line, MaterialComponent { color, texture: None, use_texture: false });
        world.set_parent(
            root,
            e_line,
            Some(Transform { position: Vec3::default(), rotation: rot }) // local: start at origin, +Z along axis
        );

        // Arrowhead: with your CUDA change (base at z=0, apex at z=+h) we place the cone’s
        // **center** at the end of the shaft (local z = axis_len), letting the kernel interpret
        // its geometry relative to that local frame.
        let e_cone = world.spawn();
        world.insert_sdf_base(e_cone, SdfBase { sdf_type: SdfType::Cone, params: [cone_h, 0.0, cone_base_r] });
        world.insert_material(e_cone, MaterialComponent { color, texture: None, use_texture: false });
        world.set_parent(
            root,
            e_cone,
            Some(Transform { position: axis_dir * axis_len, rotation: rot }) // local position at shaft end
        );

        Ok(())
    };

    // Build the 3 axes (all positive directions; no extra flips)
    build_axis(Vec3::X, rot_w_to_x, col_x)?;
    build_axis(Vec3::Y, rot_w_to_y, col_y)?;
    build_axis(Vec3::Z, rot_w_to_z, col_z)?;

    Ok(root)
}



/// Spawn a CSG: Sphere \ Difference( BoxX ∪ BoxY ∪ BoxZ )
///
/// Visually this carves 3 orthogonal slots through a sphere, like a "rounded cross".
/// Returns the entity that owns the CSG tree (material may be bound to the tree).
pub fn spawn_demo_csg(world: &mut World, _tex_mgr: &mut TextureManager) -> Result<(), Box<dyn Error>> {
    // Local lightweight error type to map CSG build errors into Box<dyn Error>
    #[derive(Debug)]
    struct SceneBuildError(String);
    impl std::fmt::Display for SceneBuildError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
    }
    impl std::error::Error for SceneBuildError {}

    // plane
    let e_plane = world.spawn();
    world.insert_transform(e_plane, Transform {
            position: Vec3::new(0.0, -3.0, 0.0),
            rotation: Quat::identity(),
        });
    world.insert_sdf_base(e_plane, SdfBase {
        sdf_type: SdfType::Plane,
        params: [0.0, 0.0, 0.0], // unused, unused, unused
    });
    // Material
    world.insert_material(e_plane, MaterialComponent {
        color: [0.5, 0.9, 0.5],
        texture: None,
        use_texture: false,
    });

    // COne
    let e_cone = world.spawn();
    world.insert_transform(e_cone, Transform {
        position: Vec3::new(1.5, 0.0, -10.0),
        rotation: Quat::identity()*Quat::from_axis_angle(Vec3::X, 90.0), // axis = world +Z (forward). Rotate if you want.
    });
    world.insert_sdf_base(e_cone, SdfBase {
        sdf_type: SdfType::Cone,
        params: [2.5, 0.0, 1.5],   // [half_height, radius_top, radius_bottom]
    });
    world.insert_material(e_cone, MaterialComponent {
        color: [0.9, 0.6, 0.3],
        texture: None,
        use_texture: false,
    });

    // Line
    let e_line = world.spawn();
    world.insert_transform(e_line, Transform {
        position: Vec3::new(1.5, -2.0, -10.0),
        rotation: Quat::identity()*Quat::from_axis_angle(Vec3::X, 90.0), // axis = world +Z (forward). Rotate if you want.
    });
    world.insert_sdf_base(e_line, SdfBase {
        sdf_type: SdfType::Line,
        params: [5.0, 0.5, 0.0],   // [length, radius, unused]
    });
    world.insert_material(e_line, MaterialComponent {
        color: [0.3, 0.6, 0.3],
        texture: None,
        use_texture: false,
    });

    // --- Parameters ---
    let center = Vec3::new(0.0, 1.0, -12.0);
    let sphere_r = 3.0f32;
    let bar = (4.8f32, 0.6f32, 0.6f32);

    // --- Leaves (SDF primitives) ---
    let e_sphere = world.spawn();
    world.insert_transform(e_sphere, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_sphere, SdfBase { sdf_type: SdfType::Sphere, params: [sphere_r, 0.0, 0.0] });
    // world.insert_rotating(e_sphere, Rotating { speed_deg_per_sec:30.0 });
    // world.insert_space_folding(e_sphere, SpaceFolding::new_3d(Mat3::Id * 20.0));

    let e_bx = world.spawn();
    world.insert_transform(e_bx, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_bx, SdfBase { sdf_type: SdfType::Cube, params: [bar.0, bar.1, bar.2] });
    // world.insert_rotating(e_bx, Rotating { speed_deg_per_sec:30.0 });

    let e_by = world.spawn();
    world.insert_transform(e_by, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_by, SdfBase { sdf_type: SdfType::Cube, params: [5.0, 5.0, 0.4] });
    // world.insert_rotating(e_by, Rotating { speed_deg_per_sec:30.0 });

    let e_bz = world.spawn();
    world.insert_transform(e_bz, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_bz, SdfBase { sdf_type: SdfType::Cube, params: [5.0, 0.4, 5.0] });
    // world.insert_rotating(e_bz, Rotating { speed_deg_per_sec:30.0 });

    // Optional distinct leaf materials (tree-level material will override if set)
    world.insert_material(e_sphere, MaterialComponent { color: [0.95, 0.92, 0.85], texture: None, use_texture: false });
    world.insert_material(e_bx,     MaterialComponent { color: [0.6, 0.1, 0.1],    texture: None, use_texture: false });
    world.insert_material(e_by,     MaterialComponent { color: [0.1, 0.6, 0.1],    texture: None, use_texture: false });
    world.insert_material(e_bz,     MaterialComponent { color: [0.1, 0.1, 0.6],    texture: None, use_texture: false });

    // --- CSG tree entity ---
    let e_tree = world.spawn();
    world.insert_material(e_tree, MaterialComponent { color: [0.85, 0.4, 0.2], texture: None, use_texture: false });
    world.insert_space_folding(e_tree, SpaceFolding::new_3d(Mat3::Id * 20.0));

    // --- Build the binary, connected CSG tree ---
    let mut tree = CsgTree::new();

    let k_s = tree.add_node(Node { node_type: NodeType::Leaf(e_sphere), parent: None, sibling: None, children: [None, None] })
        .map_err(|e| SceneBuildError(format!("add_node(sphere): {:?}", e)))?;
    let k_x = tree.add_node(Node { node_type: NodeType::Leaf(e_bx),     parent: None, sibling: None, children: [None, None] })
        .map_err(|e| SceneBuildError(format!("add_node(box X): {:?}", e)))?;
    let k_y = tree.add_node(Node { node_type: NodeType::Leaf(e_by),     parent: None, sibling: None, children: [None, None] })
        .map_err(|e| SceneBuildError(format!("add_node(box Y): {:?}", e)))?;
    let k_z = tree.add_node(Node { node_type: NodeType::Leaf(e_bz),     parent: None, sibling: None, children: [None, None] })
        .map_err(|e| SceneBuildError(format!("add_node(box Z): {:?}", e)))?;

    let k_u1 = tree.add_node(Node { node_type: NodeType::Operation(OperationType::Union),       parent: None, sibling: None, children: [None, None] })
        .map_err(|e| SceneBuildError(format!("add_node(Union u1): {:?}", e)))?;
    let k_u2 = tree.add_node(Node { node_type: NodeType::Operation(OperationType::Union),       parent: None, sibling: None, children: [None, None] })
        .map_err(|e| SceneBuildError(format!("add_node(Union u2): {:?}", e)))?;
    let k_i = tree.add_node(Node { node_type: NodeType::Operation(OperationType::Difference),  parent: None, sibling: None, children: [None, None] })
        .map_err(|e| SceneBuildError(format!("add_node(Difference): {:?}", e)))?;

    tree.connect(k_u1, k_x).map_err(|e| SceneBuildError(format!("connect(u1, bx): {:?}", e)))?;
    tree.connect(k_u1, k_y).map_err(|e| SceneBuildError(format!("connect(u1, by): {:?}", e)))?;
    tree.connect(k_u2, k_u1).map_err(|e| SceneBuildError(format!("connect(u2, u1): {:?}", e)))?;
    tree.connect(k_u2, k_z).map_err(|e| SceneBuildError(format!("connect(u2, bz): {:?}", e)))?;
    tree.connect(k_i, k_s).map_err(|e| SceneBuildError(format!("connect(df, sphere): {:?}", e)))?;
    tree.connect(k_i, k_u2).map_err(|e| SceneBuildError(format!("connect(df, union3): {:?}", e)))?;

    world.insert_csg_tree(e_tree, tree);

    //============================================================================================================================================

    //  // --- Parameters ---
    //  let center = Vec3::new(0.0, -1.0, -12.0);
    //  let sphere_r = 3.0f32;
    //  let bar = (4.8f32, 0.6f32, 0.6f32);
 
    //  // --- Leaves (SDF primitives) ---
    //  let e1_sphere = world.spawn();
    //  world.insert_transform(e1_sphere, Transform { position: center, rotation: Quat::identity() });
    //  world.insert_sdf_base(e1_sphere, SdfBase { sdf_type: SdfType::Sphere, params: [sphere_r, 0.0, 0.0] });
    //  world.insert_rotating(e1_sphere, Rotating { speed_deg_per_sec:30.0 });
 
    //  let e1_bx = world.spawn();
    //  world.insert_transform(e1_bx, Transform { position: center, rotation: Quat::identity() });
    //  world.insert_sdf_base(e1_bx, SdfBase { sdf_type: SdfType::Cube, params: [bar.0, bar.1, bar.2] });
    //  world.insert_rotating(e1_bx, Rotating { speed_deg_per_sec:30.0 });
 
    //  let e1_by = world.spawn();
    //  world.insert_transform(e1_by, Transform { position: center, rotation: Quat::identity() });
    //  world.insert_sdf_base(e1_by, SdfBase { sdf_type: SdfType::Cube, params: [bar.1, bar.0, bar.2] });
    //  world.insert_rotating(e1_by, Rotating { speed_deg_per_sec:30.0 });
 
    //  let e1_bz = world.spawn();
    //  world.insert_transform(e1_bz, Transform { position: center, rotation: Quat::identity() });
    //  world.insert_sdf_base(e1_bz, SdfBase { sdf_type: SdfType::Cube, params: [bar.2, bar.1, bar.0] });
    //  world.insert_rotating(e1_bz, Rotating { speed_deg_per_sec:30.0 });
 
    //  // Optional distinct leaf materials (tree-level material will override if set)
    //  world.insert_material(e1_sphere, MaterialComponent { color: [0.95, 0.92, 0.85], texture: None, use_texture: false });
    //  world.insert_material(e1_bx,     MaterialComponent { color: [0.6, 0.1, 0.1],    texture: None, use_texture: false });
    //  world.insert_material(e1_by,     MaterialComponent { color: [0.1, 0.6, 0.1],    texture: None, use_texture: false });
    //  world.insert_material(e1_bz,     MaterialComponent { color: [0.1, 0.1, 0.6],    texture: None, use_texture: false });
 
    //  // --- CSG tree entity ---
    //  let e1_tree = world.spawn();
    //  world.insert_material(e1_tree, MaterialComponent { color: [0.85, 0.4, 0.2], texture: None, use_texture: false });
    //  // world.insert_space_folding(e1_tree, SpaceFolding::new_3d(Mat3::Id * 20.0));
 
    //  // --- Build the binary, connected CSG tree ---
    //  let mut tree = CsgTree::new();
 
    //  let k_s = tree.add_node(Node { node_type: NodeType::Leaf(e1_sphere), parent: None, sibling: None, children: [None, None] })
    //      .map_err(|e| SceneBuildError(format!("add_node(sphere): {:?}", e)))?;
    //  let k_x = tree.add_node(Node { node_type: NodeType::Leaf(e1_bx),     parent: None, sibling: None, children: [None, None] })
    //      .map_err(|e| SceneBuildError(format!("add_node(box X): {:?}", e)))?;
    //  let k_y = tree.add_node(Node { node_type: NodeType::Leaf(e1_by),     parent: None, sibling: None, children: [None, None] })
    //      .map_err(|e| SceneBuildError(format!("add_node(box Y): {:?}", e)))?;
    //  let k_z = tree.add_node(Node { node_type: NodeType::Leaf(e1_bz),     parent: None, sibling: None, children: [None, None] })
    //      .map_err(|e| SceneBuildError(format!("add_node(box Z): {:?}", e)))?;
 
    //  let k_u1 = tree.add_node(Node { node_type: NodeType::Operation(OperationType::Union),       parent: None, sibling: None, children: [None, None] })
    //      .map_err(|e| SceneBuildError(format!("add_node(Union u1): {:?}", e)))?;
    //  let k_u2 = tree.add_node(Node { node_type: NodeType::Operation(OperationType::Union),       parent: None, sibling: None, children: [None, None] })
    //      .map_err(|e| SceneBuildError(format!("add_node(Union u2): {:?}", e)))?;
    //  let k_i = tree.add_node(Node { node_type: NodeType::Operation(OperationType::Difference),  parent: None, sibling: None, children: [None, None] })
    //      .map_err(|e| SceneBuildError(format!("add_node(Difference): {:?}", e)))?;
 
    //  tree.connect(k_u1, k_x).map_err(|e| SceneBuildError(format!("connect(u1, bx): {:?}", e)))?;
    //  tree.connect(k_u1, k_y).map_err(|e| SceneBuildError(format!("connect(u1, by): {:?}", e)))?;
    //  tree.connect(k_u2, k_u1).map_err(|e| SceneBuildError(format!("connect(u2, u1): {:?}", e)))?;
    //  tree.connect(k_u2, k_z).map_err(|e| SceneBuildError(format!("connect(u2, bz): {:?}", e)))?;
    //  tree.connect(k_i, k_s).map_err(|e| SceneBuildError(format!("connect(df, sphere): {:?}", e)))?;
    //  tree.connect(k_i, k_u2).map_err(|e| SceneBuildError(format!("connect(df, union3): {:?}", e)))?;
 
    //  world.insert_csg_tree(e1_tree, tree);
    Ok(())
}

pub fn spawn_demo_csg2(world: &mut World, _tex_mgr: &mut TextureManager) -> Result<Entity, Box<dyn Error>> {
    // Local lightweight error type to map CSG build errors into Box<dyn Error>
    #[derive(Debug)]
    struct SceneBuildError(String);
    impl std::fmt::Display for SceneBuildError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
    }
    impl std::error::Error for SceneBuildError {}

    // --- Parameters ---
    let center    = Vec3::new(0.0, 0.0, -12.0);
    let sphere_r  = 3.2_f32;                 // a bit larger to show cuts
    let bar_long  = 5.2_f32;                  // half-extent along the bar axis
    let bar_thin  = 0.32_f32;                 // half-extent across

    // --- Leaves (SDF primitives) ---
    // Big sphere (kept)
    let e_sphere = world.spawn();
    world.insert_transform(e_sphere, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_sphere, SdfBase { sdf_type: SdfType::Sphere, params: [sphere_r, 0.0, 0.0] });
    world.insert_material(e_sphere, MaterialComponent { color: [0.95, 0.92, 0.85], texture: None, use_texture: false });
    world.insert_rotating(e_sphere, Rotating { speed_deg_per_sec:30.0 });

    // Many slim bars around different axes (to grow the CSG)
    // Goal: ≥ 12–16 leaves so BnB/interval pruning becomes meaningful.
    let mut cutter_entities: Vec<Entity> = Vec::new();

    // ring around Z
    for i in 0..7 {
        let ang = (i as f32) * (std::f32::consts::PI / 8.0);
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), ang);
        let e = world.spawn();
        world.insert_transform(e, Transform { position: center, rotation: rot });
        world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cube, params: [bar_long, bar_thin, bar_thin] });
        world.insert_material(e, MaterialComponent { color: [0.25, 0.25, 0.25], texture: None, use_texture: false });
        world.insert_rotating(e, Rotating { speed_deg_per_sec:30.0 });
        cutter_entities.push(e);
    }

    // ring around Y (tilted differently)
    for i in 0..8 {
        let ang = (i as f32) * (std::f32::consts::PI / 8.0) + 0.5_f32;
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), ang);
        let e = world.spawn();
        world.insert_transform(e, Transform { position: center, rotation: rot });
        world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cube, params: [bar_long, bar_thin, bar_thin] });
        world.insert_material(e, MaterialComponent { color: [0.20, 0.20, 0.20], texture: None, use_texture: false });
        world.insert_rotating(e, Rotating { speed_deg_per_sec:30.0 });
        cutter_entities.push(e);
    }

    // --- CSG tree entity & material (tree-level can override leaf looks) ---
    let e_tree = world.spawn();
    world.insert_material(e_tree, MaterialComponent { color: [0.85, 0.4, 0.2], texture: None, use_texture: false });

    // --- Build a large binary CSG: sphere minus union(all cutters) ---
    let mut tree = CsgTree::new();

    // helper to map errors
    let mut add_leaf = |ent: Entity| -> Result<Node, SceneBuildError> {
        Ok(Node { node_type: NodeType::Leaf(ent), parent: None, sibling: None, children: [None, None] })
    };

    // 1) leaf for the sphere
    let k_s = tree
        .add_node(add_leaf(e_sphere)?)
        .map_err(|e| SceneBuildError(format!("add_node(sphere): {:?}", e)))?;

    // 2) leaves for all cutters
    let mut cutter_keys: Vec<NodeKey> = Vec::new();
    for (idx, ent) in cutter_entities.into_iter().enumerate() {
        let k = tree
            .add_node(add_leaf(ent)?)
            .map_err(|e| SceneBuildError(format!("add_node(cutter #{idx}): {:?}", e)))?;
        cutter_keys.push(k);
    }

    // 3) balanced union reduction for all cutters
    fn reduce_union(tree: &mut CsgTree, mut nodes: Vec<NodeKey>) -> Result<NodeKey, SceneBuildError> {
        while nodes.len() > 1 {
            let mut next = Vec::with_capacity((nodes.len()+1)/2);
            for pair in nodes.chunks(2) {
                if pair.len() == 1 {
                    next.push(pair[0]);
                } else {
                    let u = tree.add_node(Node { node_type: NodeType::Operation(OperationType::Union), parent: None, sibling: None, children: [None, None] })
                        .map_err(|e| SceneBuildError(format!("add_node(Union): {:?}", e)))?;
                    tree.connect(u, pair[0]).map_err(|e| SceneBuildError(format!("connect(u, a): {:?}", e)))?;
                    tree.connect(u, pair[1]).map_err(|e| SceneBuildError(format!("connect(u, b): {:?}", e)))?;
                    next.push(u);
                }
            }
            nodes = next;
        }
        Ok(nodes[0])
    }

    let k_union = reduce_union(&mut tree, cutter_keys)?;

    // 4) final Difference(sphere, union_of_cutters)
    let k_df = tree.add_node(Node { node_type: NodeType::Operation(OperationType::Intersection), parent: None, sibling: None, children: [None, None] })
        .map_err(|e| SceneBuildError(format!("add_node(Difference): {:?}", e)))?;

    tree.connect(k_df, k_s).map_err(|e| SceneBuildError(format!("connect(df, sphere): {:?}", e)))?;
    tree.connect(k_df, k_union).map_err(|e| SceneBuildError(format!("connect(df, union): {:?}", e)))?;

    // Install into the world
    world.insert_csg_tree(e_tree, tree);

    Ok(e_tree)
}

pub fn spawn_demo_csg3(world: &mut World, _tex_mgr: &mut TextureManager) -> Result<Entity, Box<dyn Error>> {
    #[derive(Debug)]
    struct SceneBuildError(String);
    impl std::fmt::Display for SceneBuildError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
    }
    impl std::error::Error for SceneBuildError {}

    // --- parameters (tweak to taste) -----------------------------------------
    let center    = Vec3::new(0.0, 0.0, -12.0);
    let sphere_r  = 4.0_f32;
    let bar_long  = 6.0_f32;   // half-extent along the bar axis
    let bar_thin  = 0.18_f32;  // half-extent across the bar

    // --- leaves ---------------------------------------------------------------
    // keep: the big sphere
    let e_sphere = world.spawn();
    world.insert_transform(e_sphere, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_sphere, SdfBase { sdf_type: SdfType::Sphere, params: [sphere_r, 0.0, 0.0] });
    world.insert_material(e_sphere, MaterialComponent { color: [0.95, 0.92, 0.85], texture: None, use_texture: false });
    world.insert_rotating(e_sphere, Rotating { speed_deg_per_sec: 20.0 });

    // 64 slender boxes as cutters:
    // - 32 in a ring around Z
    // - 32 in a ring around Y (phase shifted to avoid overlap with the Z ring)
    let mut cutters: Vec<Entity> = Vec::with_capacity(64);

    // ring around Z
    for i in 0..32 {
        let ang = (i as f32) * (2.0 * PI / 32.0);
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), ang);
        let e = world.spawn();
        world.insert_transform(e, Transform { position: center, rotation: rot });
        world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cube, params: [bar_long, bar_thin, bar_thin] });
        world.insert_material(e, MaterialComponent { color: [0.25, 0.25, 0.25], texture: None, use_texture: false });
        world.insert_rotating(e, Rotating { speed_deg_per_sec: 30.0 });
        cutters.push(e);
    }

    // ring around Y (phase shift + tilt a bit for richer intersections)
    for i in 0..31 {
        let ang = (i as f32) * (2.0 * PI / 32.0) + (PI / 32.0);
        let base = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), ang);
        // add a small extra tilt around X so cuts aren’t coplanar with the first ring
        let rot  = base * Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.20);
        let e = world.spawn();
        world.insert_transform(e, Transform { position: center, rotation: rot });
        world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cube, params: [bar_long, bar_thin, bar_thin] });
        world.insert_material(e, MaterialComponent { color: [0.22, 0.22, 0.22], texture: None, use_texture: false });
        world.insert_rotating(e, Rotating { speed_deg_per_sec: 30.0 });
        cutters.push(e);
    }

    // --- CSG owner entity (tree-level material can override leaves) ----------
    let e_tree = world.spawn();
    world.insert_material(e_tree, MaterialComponent { color: [0.85, 0.40, 0.20], texture: None, use_texture: false });
    // world.insert_space_folding(e_tree, SpaceFolding::new_3d(Mat3::Id * 20.0)); // optional

    // --- Build the large binary CSG: sphere \ union(all cutters) -------------
    let mut tree = CsgTree::new();

    // helper
    let mut add_leaf = |ent: Entity| -> Result<NodeKey, SceneBuildError> {
        tree.add_node(Node {
            node_type: NodeType::Leaf(ent),
            parent: None, sibling: None, children: [None, None]
        }).map_err(|e| SceneBuildError(format!("add_node(leaf): {:?}", e)))
    };

    // sphere leaf
    let k_sphere = add_leaf(e_sphere)?;

    // cutter leaves
    let mut cutter_keys: Vec<NodeKey> = Vec::with_capacity(cutters.len());
    for (idx, ent) in cutters.into_iter().enumerate() {
        let k = add_leaf(ent).map_err(|e| SceneBuildError(format!("add_node(cutter #{idx}): {e}")))?;
        cutter_keys.push(k);
    }

    // balanced union reduction: reduces vec of node keys to a single union node
    fn reduce_union(tree: &mut CsgTree, mut nodes: Vec<NodeKey>) -> Result<NodeKey, SceneBuildError> {
        while nodes.len() > 1 {
            let mut next = Vec::with_capacity((nodes.len() + 1) / 2);
            for pair in nodes.chunks(2) {
                if pair.len() == 1 {
                    next.push(pair[0]);
                } else {
                    let u = tree.add_node(Node {
                        node_type: NodeType::Operation(OperationType::Union),
                        parent: None, sibling: None, children: [None, None]
                    }).map_err(|e| SceneBuildError(format!("add_node(Union): {:?}", e)))?;

                    tree.connect(u, pair[0]).map_err(|e| SceneBuildError(format!("connect(union, a): {:?}", e)))?;
                    tree.connect(u, pair[1]).map_err(|e| SceneBuildError(format!("connect(union, b): {:?}", e)))?;
                    next.push(u);
                }
            }
            nodes = next;
        }
        Ok(nodes[0])
    }

    let k_union = reduce_union(&mut tree, cutter_keys)?;

    // final difference: keep sphere, subtract union of cutters
    let k_diff = tree.add_node(Node {
        node_type: NodeType::Operation(OperationType::Intersection),
        parent: None, sibling: None, children: [None, None]
    }).map_err(|e| SceneBuildError(format!("add_node(Difference): {:?}", e)))?;

    tree.connect(k_diff, k_sphere).map_err(|e| SceneBuildError(format!("connect(diff, sphere): {:?}", e)))?;
    tree.connect(k_diff, k_union ).map_err(|e| SceneBuildError(format!("connect(diff, union): {:?}", e)))?;

    // install into the world
    world.insert_csg_tree(e_tree, tree);

    Ok(e_tree)
}


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

    // Map local +Z → world +X / +Y / +Z
    let rot_w_to_x = Quat::from_axis_angle(Vec3::Y,  -90.0_f32.to_radians());  // +Z → +X
    let rot_w_to_y = Quat::from_axis_angle(Vec3::X, -90.0_f32.to_radians());  // +Z → +Y
    let rot_w_to_z = Quat::from_axis_angle(Vec3::X, 180.0_f32.to_radians());                                        // +Z → +Z

    // Build one axis entirely in gizmo-local space
    let mut build_axis = |rot: Quat, color: [f32; 3]| -> Result<(), Box<dyn std::error::Error>> {
        // Shaft: a finite line from local origin along +Z by axis_len
        let e_line = world.spawn();
        world.insert_sdf_base(e_line, SdfBase { sdf_type: SdfType::Line, params: [axis_len, shaft_radius, 0.0] });
        world.insert_material(e_line, MaterialComponent { color, texture: None, use_texture: false });
        world.set_parent(
            root,
            e_line,
            Some(Transform { position: Vec3::default(), rotation: rot })
        );

        // Cone: base at z=0, apex at z=+h (in local cone space, axis = +Z)
        // Place its local origin exactly at the shaft end, i.e. rot*(0,0,axis_len)
        let e_cone = world.spawn();
        world.insert_sdf_base(e_cone, SdfBase { sdf_type: SdfType::Cone, params: [cone_h, 0.0, cone_base_r] });
        world.insert_material(e_cone, MaterialComponent { color, texture: None, use_texture: false });
        let end_pos = rot * (Vec3::Z * axis_len); // gizmo-local position of shaft tip
        world.set_parent(
            root,
            e_cone,
            Some(Transform { position: end_pos, rotation: rot })
        );

        Ok(())
    };

    build_axis(rot_w_to_x, col_x)?;
    build_axis(rot_w_to_y, col_y)?;
    build_axis(rot_w_to_z, col_z)?;

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

pub fn spawn_blog_showcase(world: &mut World, _tex_mgr: &mut TextureManager) -> Result<Entity, Box<dyn Error>> {
    #[derive(Debug)]
    struct SceneBuildError(String);
    impl std::fmt::Display for SceneBuildError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
    }
    impl std::error::Error for SceneBuildError {}

    // Complement of the sky color defined in the CUDA path:
    // cam.image[i*3+{0,1,2}] = 0.6 * {135,206,235}
    const SKY_COMP: [f32; 3] = [
        1.0_f32 - (0.6_f32 * 135.0_f32 / 255.0_f32),
        1.0_f32 - (0.6_f32 * 206.0_f32 / 255.0_f32),
        1.0_f32 - (0.6_f32 * 235.0_f32 / 255.0_f32),
    ];

    // --- ground ---
    let e_plane = world.spawn();
    world.insert_transform(
        e_plane,
        Transform { position: Vec3::new(0.0, -3.0, 0.0), rotation: Quat::identity() }
    );
    world.insert_sdf_base(e_plane, SdfBase { sdf_type: SdfType::Plane, params: [0.0, 0.0, 0.0] });
    world.insert_material(e_plane, MaterialComponent { color: SKY_COMP, texture: None, use_texture: false });

    // --- main subject ---
    let center   = Vec3::new(0.0, 0.8, -12.0);
    let r_sphere = 4.0_f32;
    let bar_long = 6.5_f32;  // half-extent along axis
    let bar_thin = 0.18_f32; // half-extent across

    // kept: big sphere
    let e_sphere = world.spawn();
    world.insert_transform(e_sphere, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_sphere, SdfBase { sdf_type: SdfType::Sphere, params: [r_sphere, 0.0, 0.0] });
    world.insert_material(e_sphere, MaterialComponent { color: SKY_COMP, texture: None, use_texture: false });
    world.insert_rotating(e_sphere, Rotating { speed_deg_per_sec: 12.0 });

    // cutters (kept well under typical 64-leaf limits)
    let mut cutters: Vec<Entity> = Vec::new();

    // around Z
    for i in 0..18 {
        let ang = (i as f32) * (2.0 * std::f32::consts::PI / 18.0);
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), ang);
        let e = world.spawn();
        world.insert_transform(e, Transform { position: center, rotation: rot });
        world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cube, params: [bar_long, bar_thin, bar_thin] });
        world.insert_material(e, MaterialComponent { color: SKY_COMP, texture: None, use_texture: false });
        world.insert_rotating(e, Rotating { speed_deg_per_sec: 24.0 });
        cutters.push(e);
    }

    // around Y
    for i in 0..18 {
        let ang = (i as f32) * (2.0 * std::f32::consts::PI / 18.0) + 0.07;
        let rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), ang);
        let e = world.spawn();
        world.insert_transform(e, Transform { position: center, rotation: rot });
        world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cube, params: [bar_long, bar_thin, bar_thin] });
        world.insert_material(e, MaterialComponent { color: SKY_COMP, texture: None, use_texture: false });
        world.insert_rotating(e, Rotating { speed_deg_per_sec: 24.0 });
        cutters.push(e);
    }

    // around X with slight tilt
    for i in 0..16 {
        let ang = (i as f32) * (2.0 * std::f32::consts::PI / 16.0) + 0.11;
        let base = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), ang);
        let rot  = base * Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.17);
        let e = world.spawn();
        world.insert_transform(e, Transform { position: center, rotation: rot });
        world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cube, params: [bar_long, bar_thin, bar_thin] });
        world.insert_material(e, MaterialComponent { color: SKY_COMP, texture: None, use_texture: false });
        world.insert_rotating(e, Rotating { speed_deg_per_sec: 24.0 });
        cutters.push(e);
    }

    // CSG owner entity
    let e_tree = world.spawn();
    world.insert_material(e_tree, MaterialComponent { color: SKY_COMP, texture: None, use_texture: false });

    // 3D space folding: one 3D fold replaces the two 2D folds
    let fold_u = 16.0_f32; // along U
    let fold_v = 16.0_f32; // along V
    let fold_w = 28.0_f32; // along W
    let basis = Mat3::from_cols(
        Vec3::new(fold_u, 0.0,    0.0),
        Vec3::new(0.0,    fold_v, 0.0),
        Vec3::new(0.0,    0.0,    fold_w),
    );
    world.insert_space_folding(e_tree, SpaceFolding::new_3d(basis));

    // Build CSG: keep sphere ∩ union(all cutters)
    let mut tree = CsgTree::new();

    let mut add_leaf = |ent: Entity| -> Result<NodeKey, SceneBuildError> {
        tree.add_node(Node { node_type: NodeType::Leaf(ent), parent: None, sibling: None, children: [None, None] })
            .map_err(|e| SceneBuildError(format!("add_node(leaf): {:?}", e)))
    };

    let k_sphere = add_leaf(e_sphere)?;

    let mut cutter_keys: Vec<NodeKey> = Vec::with_capacity(cutters.len());
    for (idx, ent) in cutters.into_iter().enumerate() {
        let k = add_leaf(ent).map_err(|e| SceneBuildError(format!("add_node(cutter #{idx}): {e}")))?;
        cutter_keys.push(k);
    }

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
                    tree.connect(u, pair[0]).map_err(|e| SceneBuildError(format!("connect(u,a): {:?}", e)))?;
                    tree.connect(u, pair[1]).map_err(|e| SceneBuildError(format!("connect(u,b): {:?}", e)))?;
                    next.push(u);
                }
            }
            nodes = next;
        }
        Ok(nodes[0])
    }

    let k_union = reduce_union(&mut tree, cutter_keys)?;

    let k_final = tree.add_node(Node {
        node_type: NodeType::Operation(OperationType::Intersection),
        parent: None, sibling: None, children: [None, None]
    }).map_err(|e| SceneBuildError(format!("add_node(Intersection): {:?}", e)))?;
    tree.connect(k_final, k_sphere).map_err(|e| SceneBuildError(format!("connect(final,sphere): {:?}", e)))?;
    tree.connect(k_final, k_union ).map_err(|e| SceneBuildError(format!("connect(final,union): {:?}", e)))?;

    world.insert_csg_tree(e_tree, tree);

    // accents
    let cones = [
        (Vec3::new(-5.5, -1.0, -10.0), SKY_COMP),
        (Vec3::new( 5.5, -0.5, -11.5), SKY_COMP),
        (Vec3::new( 0.0,  2.5,  -8.0), SKY_COMP),
    ];
    for (pos, col) in cones {
        let e = world.spawn();
        world.insert_transform(
            e,
            Transform {
                position: pos,
                rotation: Quat::identity() * Quat::from_axis_angle(Vec3::X, 90.0),
            }
        );
        world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cone, params: [2.8, 0.0, 1.0] });
        world.insert_material(e, MaterialComponent { color: col, texture: None, use_texture: false });
    }

    Ok(e_tree)
}

pub fn spawn_menger_showcase(world: &mut World, _tex_mgr: &mut TextureManager) -> Result<Entity, Box<dyn std::error::Error>> {
    use std::cmp::min;

    #[derive(Debug)]
    struct SceneBuildError(String);
    impl std::fmt::Display for SceneBuildError { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) } }
    impl std::error::Error for SceneBuildError {}

    // --- ground plane
    let e_plane = world.spawn();
    world.insert_transform(e_plane, Transform { position: Vec3::new(0.0, -3.0, 0.0), rotation: Quat::identity() });
    world.insert_sdf_base(e_plane, SdfBase { sdf_type: SdfType::Plane, params: [0.0, 0.0, 0.0] });
    world.insert_material(e_plane, MaterialComponent { color: [0.62, 0.62, 0.65], texture: None, use_texture: false });

    // ---- parameters for the 4 cubes (levels 0..3)
    let base_y = 0.0_f32;
    let base_z = -14.0_f32;
    let spacing = 4.2_f32;
    let half_size = 1.5_f32;
    let colors = [[0.92, 0.92, 0.92]; 4];

    // cap leaves to fit your CsgTree limit (avoid MaxLeafPossibleReached)
    const MAX_HOLES: usize = 60; // tune to your builder’s max-leaf capacity

    // helpers ---------------------------------------------------------------

    // enumerate the 7 "hole" centers at one subdivision inside a parent cube
    let holes_lvl1 = |c: Vec3, h: f32| -> [(Vec3, f32); 7] {
        let step = (2.0 * h) / 3.0;         // center-to-center offset at this level
        let nh   = h / 3.0;                 // half-size of subcubes at this level
        let offs = [
            Vec3::new( 0.0,  0.0, -step),
            Vec3::new( 0.0,  0.0,  step),
            Vec3::new( 0.0, -step, 0.0 ),
            Vec3::new( 0.0,  step, 0.0 ),
            Vec3::new(-step, 0.0 , 0.0 ),
            Vec3::new( step, 0.0 , 0.0 ),
            Vec3::new( 0.0,  0.0,  0.0 ),   // center
        ];
        [
            (c + offs[0], nh), (c + offs[1], nh),
            (c + offs[2], nh), (c + offs[3], nh),
            (c + offs[4], nh), (c + offs[5], nh),
            (c + offs[6], nh),
        ]
    };

    // enumerate the 20 kept subcubes’ centers at one subdivision (for recursion)
    let kept_lvl1 = |c: Vec3, h: f32| -> Vec<(Vec3, f32)> {
        let step = (2.0 * h) / 3.0;
        let nh   = h / 3.0;
        let idx = [-1.0f32, 0.0, 1.0];
        let mut v = Vec::with_capacity(20);
        for &ix in &idx {
            for &iy in &idx {
                for &iz in &idx {
                    let ones = (ix == 0.0) as u32 + (iy == 0.0) as u32 + (iz == 0.0) as u32;
                    // keep positions with <=1 middle index (i.e., not holes)
                    if ones <= 1 {
                        v.push((c + Vec3::new(ix * step, iy * step, iz * step), nh));
                    }
                }
            }
        }
        v
    };

    // collect hole cubes up to a cap for a given recursion level
    let collect_holes = |root_center: Vec3, root_half: f32, levels: u32, cap: usize| -> Vec<(Vec3, f32)> {
        if levels == 0 { return Vec::new(); }
        let mut holes: Vec<(Vec3, f32)> = Vec::new();
        let mut frontier: Vec<(Vec3, f32)> = vec![(root_center, root_half)];
        for _lvl in 0..levels {
            // add 7 holes per kept cube
            let mut next_frontier: Vec<(Vec3, f32)> = Vec::new();
            for &(c, h) in &frontier {
                if holes.len() + 7 <= cap {
                    holes.extend_from_slice(&holes_lvl1(c, h));
                } else {
                    // partial fill if near cap
                    let batch = holes_lvl1(c, h);
                    let take = min(7, cap.saturating_sub(holes.len()));
                    holes.extend_from_slice(&batch[..take]);
                    return holes;
                }
                // enqueue the 20 kept cubes for next iteration
                next_frontier.extend(kept_lvl1(c, h));
                if holes.len() >= cap { return holes; }
            }
            frontier = next_frontier;
            if holes.len() >= cap { break; }
        }
        holes
    };

    // balanced union reduction
    fn reduce_union(tree: &mut CsgTree, mut nodes: Vec<NodeKey>) -> Result<NodeKey, SceneBuildError> {
        while nodes.len() > 1 {
            let mut next = Vec::with_capacity((nodes.len() + 1) / 2);
            for pair in nodes.chunks(2) {
                if pair.len() == 1 {
                    next.push(pair[0]);
                } else {
                    let u = tree.add_node(Node {
                        node_type: NodeType::Operation(OperationType::Union),
                        parent: None, sibling: None, children: [None, None],
                    }).map_err(|e| SceneBuildError(format!("add_node(Union): {:?}", e)))?;
                    tree.connect(u, pair[0]).map_err(|e| SceneBuildError(format!("connect(u,a): {:?}", e)))?;
                    tree.connect(u, pair[1]).map_err(|e| SceneBuildError(format!("connect(u,b): {:?}", e)))?;
                    next.push(u);
                }
            }
            nodes = next;
        }
        Ok(nodes[0])
    }

    // spawn one cube at a given recursion level (0..=3)
    let mut spawn_one = |x: f32, lvl: u32, col: [f32;3]| -> Result<(), Box<dyn std::error::Error>> {
        let center = Vec3::new(x, base_y, base_z);

        if lvl == 0 {
            // plain cube
            let e = world.spawn();
            world.insert_transform(e, Transform { position: center, rotation: Quat::identity() });
            world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cube, params: [half_size, half_size, half_size] });
            world.insert_material(e, MaterialComponent { color: col, texture: None, use_texture: false });
            return Ok(());
        }

        // build CSG: big cube \ union(holes up to cap)
        let holes = collect_holes(center, half_size, lvl, MAX_HOLES);

        let e_cube = world.spawn();
        world.insert_transform(e_cube, Transform { position: center, rotation: Quat::identity() });
        world.insert_sdf_base(e_cube, SdfBase { sdf_type: SdfType::Cube, params: [half_size, half_size, half_size] });
        world.insert_material(e_cube, MaterialComponent { color: col, texture: None, use_texture: false });

        // create hole entities
        let mut hole_entities: Vec<Entity> = Vec::with_capacity(holes.len());
        for (hc, hh) in holes {
            let eh = world.spawn();
            world.insert_transform(eh, Transform { position: hc, rotation: Quat::identity() });
            world.insert_sdf_base(eh, SdfBase { sdf_type: SdfType::Cube, params: [hh, hh, hh] });
            world.insert_material(eh, MaterialComponent { color: [0.25, 0.25, 0.25], texture: None, use_texture: false });
            hole_entities.push(eh);
        }

        // build CSG tree
        let e_tree = world.spawn();
        world.insert_material(e_tree, MaterialComponent { color: col, texture: None, use_texture: false });

        let mut tree = CsgTree::new();

        let k_keep = tree.add_node(Node {
            node_type: NodeType::Leaf(e_cube),
            parent: None, sibling: None, children: [None, None],
        }).map_err(|e| SceneBuildError(format!("add_node(keep cube): {:?}", e)))?;

        let mut hole_leaf_keys: Vec<NodeKey> = Vec::with_capacity(hole_entities.len());
        for (idx, eh) in hole_entities.into_iter().enumerate() {
            let k = tree.add_node(Node {
                node_type: NodeType::Leaf(eh),
                parent: None, sibling: None, children: [None, None],
            }).map_err(|e| SceneBuildError(format!("add_node(hole #{idx}): {:?}", e)))?;
            hole_leaf_keys.push(k);
        }

        if hole_leaf_keys.is_empty() {
            // nothing to subtract; still show the solid cube
            world.insert_csg_tree(e_tree, tree);
            return Ok(());
        }

        let k_union = reduce_union(&mut tree, hole_leaf_keys)
            .map_err(|e| SceneBuildError(format!("reduce_union: {}", e)))?;

        let k_diff = tree.add_node(Node {
            node_type: NodeType::Operation(OperationType::Difference),
            parent: None, sibling: None, children: [None, None],
        }).map_err(|e| SceneBuildError(format!("add_node(Difference): {:?}", e)))?;

        tree.connect(k_diff, k_keep).map_err(|e| SceneBuildError(format!("connect(diff, keep): {:?}", e)))?;
        tree.connect(k_diff, k_union).map_err(|e| SceneBuildError(format!("connect(diff, union): {:?}", e)))?;

        world.insert_csg_tree(e_tree, tree);
        Ok(())
    };

    // spawn four cubes like the reference image
    spawn_one(-1.5 * spacing, 0, colors[0])?;
    spawn_one(-0.5 * spacing, 1, colors[1])?;
    spawn_one( 0.5 * spacing, 2, colors[2])?; // truncated to MAX_HOLES
    spawn_one( 1.5 * spacing, 3, colors[3])?; // truncated to MAX_HOLES

    // owner group
    let root = world.spawn_group("menger_showcase_host");
    world.insert_transform(root, Transform { position: Vec3::new(0.0, 0.8, 0.0), rotation: Quat::identity() });
    Ok(root)
}

pub fn spawn_menger_showcase_host_batched(
    world: &mut World,
    _tex_mgr: &mut TextureManager,
) -> Result<Entity, Box<dyn std::error::Error>> {
    // local error wrapper
    #[derive(Debug)]
    struct SceneBuildError(String);
    impl std::fmt::Display for SceneBuildError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
    }
    impl std::error::Error for SceneBuildError {}

    // --- ground -----------------------------------------------------------------
    let e_plane = world.spawn();
    world.insert_transform(
        e_plane,
        Transform { position: Vec3::new(0.0, -3.0, 0.0), rotation: Quat::identity() },
    );
    world.insert_sdf_base(
        e_plane,
        SdfBase { sdf_type: SdfType::Plane, params: [0.0, 0.0, 0.0] },
    );
    world.insert_material(
        e_plane,
        MaterialComponent {
            // ground_plane
            color: [0.85098, 0.92941, 0.57255], // #d9ed92
            texture: None,
            use_texture: false,
        },
    );

    // --- helpers ----------------------------------------------------------------
    fn kept_lvl1(c: Vec3, h: f32) -> Vec<(Vec3, f32)> {
        let step = (2.0 * h) / 3.0;
        let nh = h / 3.0;
        let idx = [-1.0f32, 0.0, 1.0];
        let mut v = Vec::with_capacity(20);
        for &ix in &idx {
            for &iy in &idx {
                for &iz in &idx {
                    let zeros = (ix == 0.0) as u32 + (iy == 0.0) as u32 + (iz == 0.0) as u32;
                    if zeros <= 1 {
                        v.push((c + Vec3::new(ix * step, iy * step, iz * step), nh));
                    }
                }
            }
        }
        v
    }
    fn kept_positions(center: Vec3, half: f32, levels: u32) -> Vec<(Vec3, f32)> {
        if levels == 0 {
            return vec![(center, half)];
        }
        let mut frontier = vec![(center, half)];
        for _ in 0..levels {
            let mut next = Vec::new();
            for (c, h) in frontier {
                next.extend(kept_lvl1(c, h));
            }
            frontier = next;
        }
        frontier
    }
    fn reduce_union(
        tree: &mut CsgTree,
        mut nodes: Vec<NodeKey>,
    ) -> Result<NodeKey, SceneBuildError> {
        while nodes.len() > 1 {
            let mut next = Vec::with_capacity((nodes.len() + 1) / 2);
            for pair in nodes.chunks(2) {
                if pair.len() == 1 {
                    next.push(pair[0]);
                } else {
                    let u = tree
                        .add_node(Node {
                            node_type: NodeType::Operation(OperationType::Union),
                            parent: None,
                            sibling: None,
                            children: [None, None],
                        })
                        .map_err(|e| SceneBuildError(format!("add_node(Union): {:?}", e)))?;
                    tree.connect(u, pair[0])
                        .map_err(|e| SceneBuildError(format!("connect(u,a): {:?}", e)))?;
                    tree.connect(u, pair[1])
                        .map_err(|e| SceneBuildError(format!("connect(u,b): {:?}", e)))?;
                    next.push(u);
                }
            }
            nodes = next;
        }
        Ok(nodes[0])
    }
    fn spawn_union_chunk(
        world: &mut World,
        chunk: &[(Vec3, f32)],
        color: [f32; 3],
    ) -> Result<(), Box<dyn std::error::Error>> {
        match chunk.len() {
            0 => Ok(()),
            1 => {
                let (c, h) = chunk[0];
                let e = world.spawn();
                world.insert_transform(e, Transform { position: c, rotation: Quat::identity() });
                world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cube, params: [h, h, h] });
                world.insert_material(
                    e,
                    MaterialComponent { color, texture: None, use_texture: false },
                );
                Ok(())
            }
            _ => {
                let e_tree = world.spawn();
                world.insert_material(
                    e_tree,
                    MaterialComponent { color, texture: None, use_texture: false },
                );
                let mut tree = CsgTree::new();
                let mut keys: Vec<NodeKey> = Vec::with_capacity(chunk.len());
                for &(c, h) in chunk {
                    let e = world.spawn();
                    world.insert_transform(e, Transform { position: c, rotation: Quat::identity() });
                    world.insert_sdf_base(e, SdfBase { sdf_type: SdfType::Cube, params: [h, h, h] });
                    world.insert_material(
                        e,
                        MaterialComponent { color, texture: None, use_texture: false },
                    );
                    let k = tree
                        .add_node(Node {
                            node_type: NodeType::Leaf(e),
                            parent: None,
                            sibling: None,
                            children: [None, None],
                        })
                        .map_err(|e| SceneBuildError(format!("add_node(leaf): {:?}", e)))?;
                    keys.push(k);
                }
                let _ = reduce_union(&mut tree, keys)?;
                world.insert_csg_tree(e_tree, tree);
                Ok(())
            }
        }
    }

    // --- layout -----------------------------------------------------------------
    let centers = [
        Vec3::new(-6.3, 0.0, -14.0), // L0
        Vec3::new(-2.1, 0.0, -14.0), // L1
        Vec3::new( 2.1, 0.0, -14.0), // L2
        Vec3::new( 6.3, 0.0, -14.0), // L3
    ];
    let half = 1.5_f32;

    // colors per level
    let level_colors: [[f32; 3]; 4] = [
        [118.0/255.0, 200.0/255.0, 147.0/255.0], // L0 = #76c893
        [153.0/255.0, 217.0/255.0, 140.0/255.0], // L1 = #99d98c
        [181.0/255.0, 228.0/255.0, 140.0/255.0], // L2 = #b5e48c
        [217.0/255.0, 237.0/255.0, 146.0/255.0], // L3 = #d9ed92
    ];


    // cap per CSG tree to stay under MAX_LEAFS (64)
    const PER_TREE_CAP: usize = 60;

    for (lvl, center) in centers.iter().enumerate() {
        let leaves = kept_positions(*center, half, lvl as u32);
        let col = level_colors[lvl];
        for chunk in leaves.chunks(PER_TREE_CAP) {
            spawn_union_chunk(world, chunk, col)?;
        }
    }

    // optional root object to move whole showcase later
    let root = world.spawn_group("menger_host_batched");
    world.insert_transform(
        root,
        Transform { position: Vec3::new(0.0, 0.0, 0.0), rotation: Quat::identity() },
    );
    Ok(root)
}


use std::error::Error;

use crate::ecs::ecs::{World, Entity};
use crate::ecs::ecs::components::{
    Transform, MaterialComponent, SdfBase, SdfType, Rotating,
    TextureManager,
    CsgTree, Node, NodeType, OperationType,
};
use crate::ecs::ecs::{Vec3, Quat};

/// Spawn a CSG: Sphere \ Difference( BoxX ∪ BoxY ∪ BoxZ )
///
/// Visually this carves 3 orthogonal slots through a sphere, like a "rounded cross".
/// Returns the entity that owns the CSG tree (material may be bound to the tree).
pub fn spawn_demo_csg(world: &mut World, _tex_mgr: &mut TextureManager) -> Result<Entity, Box<dyn Error>> {
    // Local lightweight error type to map CSG build errors into Box<dyn Error>
    #[derive(Debug)]
    struct SceneBuildError(String);
    impl std::fmt::Display for SceneBuildError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
    }
    impl std::error::Error for SceneBuildError {}

    // --- Parameters ---
    let center = Vec3::new(0.0, 0.0, -12.0);
    let sphere_r = 3.0f32;
    let bar = (4.8f32, 0.6f32, 0.6f32);

    // --- Leaves (SDF primitives) ---
    let e_sphere = world.spawn();
    world.insert_transform(e_sphere, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_sphere, SdfBase { sdf_type: SdfType::Sphere, params: [sphere_r, 0.0, 0.0] });
    world.insert_rotating(e_sphere, Rotating { speed_deg_per_sec:30.0 });

    let e_bx = world.spawn();
    world.insert_transform(e_bx, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_bx, SdfBase { sdf_type: SdfType::Cube, params: [bar.0, bar.1, bar.2] });
    world.insert_rotating(e_bx, Rotating { speed_deg_per_sec:30.0 });

    let e_by = world.spawn();
    world.insert_transform(e_by, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_by, SdfBase { sdf_type: SdfType::Cube, params: [bar.1, bar.0, bar.2] });
    world.insert_rotating(e_by, Rotating { speed_deg_per_sec:30.0 });

    let e_bz = world.spawn();
    world.insert_transform(e_bz, Transform { position: center, rotation: Quat::identity() });
    world.insert_sdf_base(e_bz, SdfBase { sdf_type: SdfType::Cube, params: [bar.2, bar.1, bar.0] });
    world.insert_rotating(e_bz, Rotating { speed_deg_per_sec:30.0 });

    // Optional distinct leaf materials (tree-level material will override if set)
    world.insert_material(e_sphere, MaterialComponent { color: [0.95, 0.92, 0.85], texture: None, use_texture: false });
    world.insert_material(e_bx,     MaterialComponent { color: [0.6, 0.1, 0.1],    texture: None, use_texture: false });
    world.insert_material(e_by,     MaterialComponent { color: [0.1, 0.6, 0.1],    texture: None, use_texture: false });
    world.insert_material(e_bz,     MaterialComponent { color: [0.1, 0.1, 0.6],    texture: None, use_texture: false });

    // --- CSG tree entity ---
    let e_tree = world.spawn();
    world.insert_material(e_tree, MaterialComponent { color: [0.85, 0.4, 0.2], texture: None, use_texture: false });

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

    Ok(e_tree)
}

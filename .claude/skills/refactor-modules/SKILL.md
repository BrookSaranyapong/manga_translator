# refactor-modules

Review module structure, identify tight coupling, and implement the target architecture.

## Usage

1. List all Python files under `modules/`
2. Identify issues: tight coupling, god modules, flat namespace, orphan imports
3. Compare against the [target architecture](references/target-architecture.md)
4. If files don't match target structure, propose and implement moves with `git mv`
5. Update all imports to match new paths
6. Verify with `pyright` that no broken references were introduced

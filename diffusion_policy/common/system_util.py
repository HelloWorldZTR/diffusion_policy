def raise_data_loading_resource_limits():
    """Raise process resource limits used by DataLoader IPC where possible."""
    try:
        import resource
    except ImportError:
        return

    if hasattr(resource, "RLIMIT_NOFILE"):
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard > soft:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            except (ValueError, OSError):
                pass

    if hasattr(resource, "RLIMIT_MEMLOCK"):
        _, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        try:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
        except (ValueError, OSError):
            pass


def set_torch_file_system_sharing_strategy():
    """Use tmpfile-backed tensor sharing to avoid fd-based shm pressure."""
    try:
        import torch.multiprocessing as torch_mp
    except ImportError:
        return

    try:
        torch_mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass


def configure_data_loading_process():
    raise_data_loading_resource_limits()
    set_torch_file_system_sharing_strategy()


def data_loader_worker_init_fn(_worker_id):
    configure_data_loading_process()

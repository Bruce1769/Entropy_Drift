from types import SimpleNamespace

from r2r.models.sglang_patch.slm_server import SLMServer


def test_cache_finished_req_compat_uses_root_node_when_last_node_is_missing():
    root_node = object()
    seen = {}

    class FakeTreeCache:
        def __init__(self):
            self.root_node = root_node

        def cache_finished_req(self, req):
            seen["last_node"] = req.last_node
            seen["prefix_indices"] = req.prefix_indices

    scheduler = SimpleNamespace(tree_cache=FakeTreeCache())
    req = SimpleNamespace(rid="req-1", last_node=None, prefix_indices=None)

    SLMServer.cache_finished_req_compat(rank=0, scheduler=scheduler, req=req)

    assert req.last_node is root_node
    assert req.prefix_indices == []
    assert seen["last_node"] is root_node
    assert seen["prefix_indices"] == []


def test_cache_finished_req_compat_can_resolve_root_from_match_prefix():
    root_node = object()
    seen = {}

    class FakeTreeCache:
        def match_prefix(self, key):
            assert key == []
            return SimpleNamespace(last_device_node=root_node)

        def cache_finished_req(self, req):
            seen["last_node"] = req.last_node

    scheduler = SimpleNamespace(tree_cache=FakeTreeCache())
    req = SimpleNamespace(rid="req-2", last_node=None, prefix_indices=[])

    SLMServer.cache_finished_req_compat(rank=0, scheduler=scheduler, req=req)

    assert req.last_node is root_node
    assert seen["last_node"] is root_node

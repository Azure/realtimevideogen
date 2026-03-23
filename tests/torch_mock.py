"""
A mock for the torch module with specific mocked attributes and methods for testing purposes.
"""

from unittest.mock import MagicMock

import importlib.util

from typing import Callable
from typing import Tuple
from typing import Dict
from typing import Any
from typing import List
from typing import Optional


class FakeModule:
    def __init__(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        pass


class FakeLayerNorm(FakeModule):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)


"""
class FakeFunctional(MagicMock):
    pass


class FakeNN:
    Module = FakeModule

    def __getattr__(self, name):
        cls = type(name, (FakeModule,), {})
        setattr(self, name, cls)
        return cls
"""


class TorchMock(MagicMock):
    """A mock for the torch module with specific mocked attributes and methods."""

    def __init__(
        self,
        *args: Tuple,
        **kwargs: Dict
    ) -> None:
        super().__init__(*args, **kwargs)

        self.__spec__ = importlib.util.spec_from_loader("torch", loader=None)

        # self.nn = FakeNN()

        # Define real exception class for torch.OutOfMemoryError
        self.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})

        # Mock torch.cuda.memory_allocated
        self.cuda = MagicMock()
        self.cuda.memory_allocated.return_value = 0
        self.cuda.device_count.return_value = 1
        self.cuda.get_device_name.side_effect = lambda device=None: f"MockDevice:{device if device is not None else 0}"
        self.cuda.synchronize = lambda device=None: None

        # Mock @torch.inference_mode() decorator
        self.inference_mode = lambda: self._noop_decorator

        # Mock tensor creation functions
        self.randn = lambda *shape, **kwargs: self._mock_tensor(*shape)
        self.ones = lambda *shape, **kwargs: self._mock_tensor(*shape, fill_value=1)
        self.tensor = lambda data, **kwargs: self._mock_tensor_from_data(data)

        # Mock load/save
        self._save_store: Dict[Any, Any] = {}  # internal dict to track saved "objects"

        def mock_save(
            obj: Any,
            f: Any,
            **kwargs: Dict[str, Any]
        ) -> None:
            self._save_store[f] = obj
            return None

        def mock_load(
            f: Any,
            map_location: Optional[Any] = None,
            **kwargs: Dict[str, Any]
        ) -> Any:
            return self._save_store.get(f, MagicMock(name="LoadedTensor"))

        self.save = mock_save
        self.load = mock_load

    @staticmethod
    def _noop_decorator(func: Callable) -> Callable:
        return func

    @staticmethod
    def _mock_tensor(*shape: int, fill_value: Any = 0) -> MagicMock:
        m = MagicMock()
        m.shape = shape
        m.ndim = len(shape)
        m.fill_value = fill_value

        def chunk(chunks: int, dim: int = 0) -> List[Any]:
            return [m for _ in range(chunks)]

        m.chunk.side_effect = chunk
        m.__getitem__.side_effect = lambda idx: fill_value
        return m

    @staticmethod
    def _mock_tensor_from_data(data: Any) -> MagicMock:
        """Return a MagicMock representing a tensor from given data."""
        m = MagicMock()
        if hasattr(data, "__len__"):
            m.shape = (len(data),)
        else:
            m.shape = ()
        m.data = data
        m.ndim = len(m.shape)
        return m

    def get_sub_modules(self) -> Dict[str, Any]:
        return {
            "torch": self,
            "torch.fft": MagicMock(),
            "torch.nn": MagicMock(),
            "torch.nn.functional": MagicMock(),
            # "torch.nn": self.nn,
            # "torch.nn.Module": FakeModule,
            # "torch.nn.LayerNorm": FakeLayerNorm,
            # "torch.nn.functional": self.nn.functional,
            "torch.nn.parallel": MagicMock(),
            "torch.nn.parallel.distributed": MagicMock(),
            "torch.nn.modules": MagicMock(),
            "torch.nn.modules.utils": MagicMock(),
            "torch.nn.common_types": MagicMock(),
            "torch.amp": MagicMock(),
            "torch.amp.grad_scaler": MagicMock(),
            "torch.distributed": MagicMock(),
            "torch.distributed.rpc": MagicMock(),
            "torch.distributed.algorithms": MagicMock(),
            "torch.distributed.algorithms.join": MagicMock(),
            "torch.optim": MagicMock(),
            "torch.optim.lr_scheduler": MagicMock(),
            "torch.utils": MagicMock(),
            "torch.utils.data": MagicMock(),
            "torch.utils.hooks": MagicMock(),
            "torch.utils.checkpoint": MagicMock(),
            "torch.utils.serialization": MagicMock(),
            "torch.utils.model_zoo": MagicMock(),
            "torch.utils._ordered_set": MagicMock(),
            "torch.utils._sympy": MagicMock(),
            "torch.utils._sympy.functions": MagicMock(),
            "torch.utils._pytree": MagicMock(),
            "torch.cuda": MagicMock(),
        }

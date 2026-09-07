#
# CVLinearWrapper - Splits a Linear layer into quantize(Vector) + matmul(Cube)
#
import torch

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)


class CVLinearWrapper:
    """
    Splits a Linear layer into quantize(Vector) + matmul(Cube).

    Automatically detects TP communication operations:
    - No communication (ReplicatedLinear): W8A8 is split into independent quantize + matmul
    - Has communication (ColumnParallelLinear with custom_op): automatically falls back to full forward

    Usage example:
        wrapper = CVLinearWrapper(linear)

        # Step 1: Quantize (Vector)
        q_quant, q_scale = wrapper.quantize(x)

        # Step 2: Matrix multiply (Cube)
        result = wrapper.matmul(q_quant, q_scale)
    """

    # Schemes whose dynamic activation quantization can run as a standalone
    # Vector op, decoupled from the Cube matmul. Their apply() accepts a
    # pre-quantized (quantized_x, pertoken_scale) tuple and skips the
    # internal quant on the matmul side.
    _SPLITTABLE_SCHEMES = (
        AscendW8A8DynamicLinearMethod,  # also covers the W8A8FP8_DYNAMIC subclass
        AscendW8A8MXFP8DynamicLinearMethod,  # also covers the DS block-quant subclass
    )

    def __init__(self, linear):
        self.linear = linear

        # Detect whether TP communication operations exist
        self._has_communication = self._detect_communication(linear)

        # Detect quantization scheme
        # Handles two cases:
        # 1. linear.quant_method is directly the scheme instance
        # 2. linear.quant_method is a wrapper class, requiring .quant_method
        #    to get the actual quantization method
        inner = getattr(linear.quant_method, "quant_method", linear.quant_method)

        # One predicate decides whether the layer can be split at all; the
        # scheme differences are just parameters (act dtype / mxfp flag) fed
        # to the device quant op, so no per-scheme branches are needed here.
        self._splittable = not self._has_communication and isinstance(inner, self._SPLITTABLE_SCHEMES)
        self._use_mxfp_quant = isinstance(inner, AscendW8A8MXFP8DynamicLinearMethod)
        # W8A8-family schemes carry the activation dtype (int8 / fp8) on the
        # scheme; MXFP8 always targets e4m3, which is the operator default.
        self._act_quant_type = getattr(inner, "act_quant_type", None) or torch.float8_e4m3fn
        # Two wrappers with equal keys produce identical quantize() outputs on
        # the same input, so one call may serve both (see share_quant users).
        self.quant_share_key = (
            self._splittable,
            self._has_communication,
            self._use_mxfp_quant,
            self._act_quant_type,
        )

    @property
    def splittable(self) -> bool:
        """Whether quantize() performs a real dynamic quant (vs pass-through)."""
        return self._splittable

    @staticmethod
    def _detect_communication(linear):
        """
        Detect whether the Linear layer has TP communication during forward.

        Criteria:
        - custom_op is None or CustomReplicatedOp: no TP communication
        - Other custom_op (e.g., MLPColumnParallelOp with all_gather): has TP communication
        - ColumnParallelLinear with gather_output=True: has all-gather communication
        Note: ColumnParallelLinear even with custom_op=None only communicates when gather_output=True.
              wq_b uses default gather_output=False, so no communication and can be split.
        """
        custom_op = getattr(linear, "custom_op", None)
        if custom_op is not None:
            from vllm_ascend.ops.linear_op import CustomReplicatedOp

            if not isinstance(custom_op, CustomReplicatedOp):
                return True

        return hasattr(linear, "gather_output") and linear.gather_output

    def quantize(self, x: torch.Tensor):
        """
        Execute only the quantization step (Vector operator).

        Args:
            x: Input tensor

        Returns:
            (quantized_x, pertoken_scale): Quantized tensor and scaling factor.
            For linear layers with communication or without a splittable
            dynamic-quant scheme, returns (x, None) unchanged.
        """
        if not self._splittable:
            return x, None

        # The device adaptor dispatches on these parameters: W8A8 goes to
        # npu_dynamic_quant(dst_type=act_quant_type) and MXFP8 to
        # npu_dynamic_mx_quant. The MXFP8 scale_alg is resolved inside the
        # device op from the global config (equivalent to the scheme's
        # instance-frozen value: both derive from the same model config).
        return DeviceOperator.npu_dynamic_quant(
            x,
            act_quant_type=self._act_quant_type,
            use_mxfp_quant=self._use_mxfp_quant,
        )

    def matmul(self, quantized_x: torch.Tensor, pertoken_scale=None, bias=None):
        """
        Execute only the matrix multiplication step (Cube operator).

        Args:
            quantized_x: Quantized input (original input when communication is present)
            pertoken_scale: Per-token scaling factor for W8A8_DYNAMIC
            bias: Bias

        Returns:
            Matrix multiplication result
        """
        if self._has_communication:
            return self.linear.forward(quantized_x)

        # A non-None scale marks the input as already quantized: forward it
        # as a (tensor, scale) tuple so the scheme's apply() skips its
        # internal quant. Unquantized input reaches apply() unchanged.
        if pertoken_scale is not None:
            return self.linear.quant_method.apply(self.linear, (quantized_x, pertoken_scale), bias)
        return self.linear.quant_method.apply(self.linear, quantized_x, bias)

    def forward(self, x: torch.Tensor, bias=None):
        """Full forward (equivalent to the original Linear.forward)"""
        q_quant, q_scale = self.quantize(x)
        return self.matmul(q_quant, q_scale, bias)

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        self.linear.weight = value

    def __getattr__(self, name):
        """Delegate undefined attributes to the inner linear object"""
        return getattr(self.linear, name)

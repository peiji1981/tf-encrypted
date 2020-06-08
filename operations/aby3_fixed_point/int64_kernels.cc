#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
using namespace tensorflow;

// Some missing operations in Tensorflow.
using u32 = tensorflow::uint32;
using u64 = tensorflow::uint64;
using i64 = tensorflow::int64;

/// BitReverse uint64
class I64BitReverseOp : public OpKernel {
 public:
  explicit I64BitReverseOp(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *ctx) override {
    const Tensor &op0 = ctx->input(0);
    Tensor *output;
    TensorShape out_shape{op0.shape()};
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    auto u32_reverse = [](u32 v) -> u32 {
      v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
      // swap consecutive pairs
      v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
      // swap nibbles ...
      v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
      // swap bytes
      v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
      // swap 2-byte long pairs
      v = (v >> 16) | (v << 16);
      return v;
    };

    const u64 *src = (const u64 *)op0.flat<i64>().data();
    const u64 *end = src + op0.NumElements();
    i64 *dst = output->flat<i64>().data();
    std::transform(src, end, dst, [u32_reverse](u64 v) -> i64 {
      u32 *raw = (u32 *)&v;
      raw[0] = u32_reverse(raw[0]);
      raw[1] = u32_reverse(raw[1]);
      std::swap(raw[0], raw[1]);
      return (i64)v;
    });
  }
};

/// Gather bits of even (or odd) positions.
/// For example Gather((b0, b1, b2, b3)_2, even = True) => (b0, b2)_2
///             Gather((b0, b1, b2, b3)_2, even = False) => (b1, b3)_2
class I64BitGatherOp : public OpKernel {
 public:
  explicit I64BitGatherOp(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *ctx) override {
    const Tensor &op0 = ctx->input(0);
    bool even = ctx->input(1).scalar<bool>()();
    Tensor *output;
    TensorShape out_shape{op0.shape()};
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    const u64 *src = (const u64 *)op0.flat<i64>().data();
    const u64 *end = src + op0.NumElements();
    i64 *dst = output->flat<i64>().data();
    std::transform(src, end, dst, [even](u64 v) -> i64 {
      const u64 one{1};
      u64 ans{0};
      if (!even) v >>= 1;
      for (long d = 0; v > 0 && d < 32; ++d) {
        if (v & 1) ans |= (one << d);
        v >>= 2;
      }
      return ans;
    });
  }
};

/// XOR the positions with non-zero.
/// For bit sequence, b0,b1,b2,...,b_63, compute XOR_{i: b_i = 1}(i)
class I64XorIndicesOp : public OpKernel {
 public:
  explicit I64XorIndicesOp(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *ctx) override {
    const Tensor &op0 = ctx->input(0);
    Tensor *output;
    TensorShape out_shape{op0.shape()};
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    const u64 *src = (const u64 *)op0.flat<i64>().data();
    const u64 *end = src + op0.NumElements();
    u64 *dst = (u64 *)output->flat<i64>().data();
    std::transform(src, end, dst, [](u64 v) -> u64 {
      long ans{0};
      for (long d = 0; v > 0 && d < 64; ++d) {
        if (v & 1) ans ^= d;
        v >>= 1;
      }
      return (u64)ans;
    });
  }
};

REGISTER_OP("I64BitReverse").Input("op0: int64").Output("output: int64").SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("I64BitGather")
    .Input("op0: int64")
    .Input("op1: bool")
    .Output("output: int64")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("I64XorIndices").Input("op0: int64").Output("output: int64").SetShapeFn(shape_inference::UnchangedShape);

REGISTER_KERNEL_BUILDER(Name("I64BitReverse").Device(DEVICE_CPU), I64BitReverseOp);
REGISTER_KERNEL_BUILDER(Name("I64BitGather").Device(DEVICE_CPU), I64BitGatherOp);
REGISTER_KERNEL_BUILDER(Name("I64XorIndices").Device(DEVICE_CPU), I64XorIndicesOp);

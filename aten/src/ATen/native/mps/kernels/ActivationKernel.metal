#include <ATen/native/mps/kernels/Activation.h>
#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

struct hardshrink_functor {
  template <typename T>
  inline T operator()(const T x, const T lambda) {
    return abs(float(x)) <= float(lambda) ? T(0) : x;
  }
};

struct softshrink_functor {
  template <typename T>
  inline T operator()(const T x, const T lambda) {
    if (x > lambda) {
      return x - lambda;
    } else if (x < -lambda) {
      return x + lambda;
    } else {
      return T(0);
    }
  }
};

struct shrink_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T x, const T lambda) {
    return abs(float(x)) <= float(lambda) ? T(0) : grad_output;
  }
};

REGISTER_UNARY_ALPHA_OP(hardshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(hardshrink, half, half, half);
REGISTER_UNARY_ALPHA_OP(hardshrink, bfloat, bfloat, bfloat);

REGISTER_UNARY_ALPHA_OP(softshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(softshrink, half, half, half);
REGISTER_UNARY_ALPHA_OP(softshrink, bfloat, bfloat, bfloat);

REGISTER_BINARY_ALPHA_OP(shrink_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(shrink_backward, half, half, half);
REGISTER_BINARY_ALPHA_OP(shrink_backward, bfloat, bfloat, bfloat);

struct relu_functor {
  template <typename T>
  inline T operator()(const T x) {
    return x > T(0) ? x : T(0);
  }
};

REGISTER_UNARY_OP(relu, float, float);
REGISTER_UNARY_OP(relu, half, half);
REGISTER_UNARY_OP(relu, bfloat, bfloat);
REGISTER_UNARY_OP(relu, long, long);
REGISTER_UNARY_OP(relu, int, int);
REGISTER_UNARY_OP(relu, short, short);
REGISTER_UNARY_OP(relu, char, char);
REGISTER_UNARY_OP(relu, uchar, uchar);
REGISTER_UNARY_OP(relu, bool, bool);

struct hardsigmoid_functor {
  template <typename T>
  inline T operator()(const T x) {
    return static_cast<T>(min(max(x + 3.0f, .0f), 6.f) / 6.f);
  }
};

struct hardsigmoid_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    constexpr auto one_sixth = 1.0f / 6.0f;
    return static_cast<T>(
        abs(float(self)) < 3.0f ? float(grad_output) * one_sixth : 0.0f);
  }
};

REGISTER_UNARY_OP(hardsigmoid, float, float);
REGISTER_UNARY_OP(hardsigmoid, half, half);
REGISTER_UNARY_OP(hardsigmoid, bfloat, bfloat);

REGISTER_BINARY_OP(hardsigmoid_backward, float, float);
REGISTER_BINARY_OP(hardsigmoid_backward, half, half);
REGISTER_BINARY_OP(hardsigmoid_backward, bfloat, bfloat);

struct hardswish_functor {
  template <typename T>
  inline T operator()(const T x) {
    return static_cast<T>(float(x) * min(max(float(x) + 3.0f, .0f), 6.f) / 6.f);
  }
};

struct hardswish_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    constexpr T zero(0);
    constexpr T three(3);
    constexpr T neg_three(-3);

    if (self <= neg_three) {
      return zero;
    } else if (self >= three) {
      return grad_output;
    } else {
      return static_cast<T>(float(grad_output) * (float(self) / 3.0f + 0.5f));
    }
  }
};

REGISTER_UNARY_OP(hardswish, float, float);
REGISTER_UNARY_OP(hardswish, half, half);
REGISTER_UNARY_OP(hardswish, bfloat, bfloat);

REGISTER_BINARY_OP(hardswish_backward, float, float);
REGISTER_BINARY_OP(hardswish_backward, half, half);
REGISTER_BINARY_OP(hardswish_backward, bfloat, bfloat);

struct elu_functor {
  template <typename T>
  inline T operator()(const T self_, const ELUParams<T> params) {
    using op_T = opmath_t<T>;
    auto alpha = static_cast<op_T>(params.alpha);
    auto scale = static_cast<op_T>(params.scale);
    auto input_scale = static_cast<op_T>(params.input_scale);
    auto self = static_cast<op_T>(self_);
    auto neg_res = alpha * (::metal::precise::exp(self * input_scale) - 1);
    return static_cast<T>(scale * (self < 0 ? neg_res : self));
  }
};

struct elu_backward_functor {
  template <typename T>
  inline T operator()(
      const T grad_output_,
      const T self_,
      ELUBackwardParams<T> params) {
    using op_T = opmath_t<T>;
    auto alpha = static_cast<op_T>(params.alpha);
    auto scale = static_cast<op_T>(params.scale);
    auto input_scale = static_cast<op_T>(params.input_scale);
    auto grad_output = static_cast<op_T>(grad_output_);
    auto self = static_cast<op_T>(self_);

    if (params.is_result) {
      auto neg_coef = input_scale * (self + alpha * scale);
      return static_cast<T>(grad_output * (self <= 0 ? neg_coef : scale));
    } else {
      auto neg_coef = input_scale * alpha * scale *
          ::metal::precise::exp(self * input_scale);
      return static_cast<T>(grad_output * (self <= 0 ? neg_coef : scale));
    }
  }
};

#define REGISTER_ELU_OP(T)            \
  typedef ELUParams<T> ELUParams_##T; \
  REGISTER_UNARY_ALPHA_OP(elu, T, ELUParams_##T, T);

REGISTER_ELU_OP(float);
REGISTER_ELU_OP(half);
REGISTER_ELU_OP(bfloat);

#define REGISTER_ELU_BACKWARD_OP(T)                   \
  typedef ELUBackwardParams<T> ELUBackwardParams_##T; \
  REGISTER_BINARY_ALPHA_OP(elu_backward, T, ELUBackwardParams_##T, T);

REGISTER_ELU_BACKWARD_OP(float);
REGISTER_ELU_BACKWARD_OP(half);
REGISTER_ELU_BACKWARD_OP(bfloat);

struct leaky_relu_functor {
  template <typename T>
  inline T operator()(const T x, const T negative_slope) {
    return float(x) > 0.0f ? x
                           : static_cast<T>(float(x) * float(negative_slope));
  }
};

struct leaky_relu_backward_functor {
  template <typename T>
  inline T operator()(
      const T self,
      const T grad_output,
      const T negative_slope) {
    return float(self) > 0.0f
        ? grad_output
        : static_cast<T>(float(grad_output) * float(negative_slope));
  }
};

REGISTER_UNARY_ALPHA_OP(leaky_relu, float, float, float);
REGISTER_UNARY_ALPHA_OP(leaky_relu, half, half, half);
REGISTER_UNARY_ALPHA_OP(leaky_relu, bfloat, bfloat, bfloat);

REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, half, half, half);
REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, bfloat, bfloat, bfloat);

struct silu_functor {
  template <typename T>
  inline T operator()(const T x) {
    float xf = float(x);
    return static_cast<T>(xf / (1.0f + ::metal::precise::exp(-xf)));
  }
};

REGISTER_UNARY_OP(silu, float, float);
REGISTER_UNARY_OP(silu, half, half);
REGISTER_UNARY_OP(silu, bfloat, bfloat);
REGISTER_UNARY_OP(silu, int, int);
REGISTER_UNARY_OP(silu, short, short);
REGISTER_UNARY_OP(silu, char, char);
REGISTER_UNARY_OP(silu, uchar, uchar);
REGISTER_UNARY_OP(silu, bool, bool);

struct silu_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    float sf = float(self);
    float sig = 1.0f / (1.0f + ::metal::precise::exp(-sf));
    return static_cast<T>(float(grad_output) * sig * (1.0f + sf - sf * sig));
  }
};

REGISTER_BINARY_OP(silu_backward, float, float);
REGISTER_BINARY_OP(silu_backward, half, half);
REGISTER_BINARY_OP(silu_backward, bfloat, bfloat);

// ================================================================
//  GELU forward and backward kernels
//  All math done in float32 regardless of input dtype for precision.
//
//  Exact (none):
//    fwd:  x * 0.5 * (1 + erf(x / sqrt(2)))
//    bwd:  dy * (cdf(x) + x * pdf(x))
//
//  Tanh approximation:
//    fwd:  0.5 * x * (1 + tanh(beta*(x + kappa*x^3)))
//    bwd:  dy * (0.5*(1+t) + 0.5*x*(1-t^2)*beta*(1+3*kappa*x^2))
//          where t = tanh(beta*(x+kappa*x^3))
// ================================================================

struct gelu_none_functor {
  template <typename T>
  inline T operator()(const T x) {
    constexpr float kAlpha = 0.7071067811865476f; // 1/sqrt(2)
    float xf = float(x);
    return static_cast<T>(xf * 0.5f * (1.0f + erf(xf * kAlpha)));
  }
};

REGISTER_UNARY_OP(gelu_none, float, float);
REGISTER_UNARY_OP(gelu_none, half, half);
REGISTER_UNARY_OP(gelu_none, bfloat, bfloat);

struct gelu_tanh_functor {
  template <typename T>
  inline T operator()(const T x) {
    constexpr float kBeta = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kKappa = 0.044715f;
    float xf = float(x);
    float inner = kBeta * (xf + kKappa * xf * xf * xf);
    return static_cast<T>(0.5f * xf * (1.0f + tanh(inner)));
  }
};

REGISTER_UNARY_OP(gelu_tanh, float, float);
REGISTER_UNARY_OP(gelu_tanh, half, half);
REGISTER_UNARY_OP(gelu_tanh, bfloat, bfloat);

struct gelu_backward_none_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T x) {
    constexpr float kAlpha = 0.7071067811865476f; // 1/sqrt(2)
    constexpr float kBeta = 0.3989422804014327f; // 1/sqrt(2*pi)
    float xf = float(x);
    float cdf = 0.5f * (1.0f + erf(xf * kAlpha));
    float pdf = exp(-0.5f * xf * xf) * kBeta;
    return static_cast<T>(float(grad_output) * (cdf + xf * pdf));
  }
};

REGISTER_BINARY_OP(gelu_backward_none, float, float);
REGISTER_BINARY_OP(gelu_backward_none, half, half);
REGISTER_BINARY_OP(gelu_backward_none, bfloat, bfloat);

struct gelu_backward_tanh_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T x) {
    constexpr float kBeta = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kKappa = 0.044715f;
    float xf = float(x);
    float inner = kBeta * (xf + kKappa * xf * xf * xf);
    float t = tanh(inner);
    float left_d = 0.5f * (1.0f + t);
    float right_d =
        0.5f * xf * (1.0f - t * t) * kBeta * (1.0f + 3.0f * kKappa * xf * xf);
    return static_cast<T>(float(grad_output) * (left_d + right_d));
  }
};

REGISTER_BINARY_OP(gelu_backward_tanh, float, float);
REGISTER_BINARY_OP(gelu_backward_tanh, half, half);
REGISTER_BINARY_OP(gelu_backward_tanh, bfloat, bfloat);

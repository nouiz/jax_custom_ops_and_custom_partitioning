import os
# os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_proto --xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_to=custom_part_ln_dump"

from functools import partial

import numpy as np
import jax
from jax import random
from jax import core, vmap
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental.custom_partitioning import custom_partitioning
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec, NamedSharding
P = PartitionSpec

from collections.abc import Sequence

# ------------------------------------------------------------------------
from transformer_engine.jax.cpp_extensions import _layernorm_fwd_p, _layernorm_bwd_p
def te_layernorm_fwd(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                     zero_centered_gamma: bool, epsilon: float):
    return _layernorm_fwd_p.bind(x, gamma, beta,
                                 zero_centered_gamma=zero_centered_gamma,
                                 epsilon=epsilon)

def te_layernorm_bwd(g: jnp.ndarray, mu: jnp.ndarray, rsigma: jnp.ndarray, x: jnp.ndarray,
                     gamma: jnp.ndarray, zero_centered_gamma: bool, epsilon: float):
    return _layernorm_bwd_p.bind(g, mu, rsigma, x, gamma,
                                 zero_centered_gamma=zero_centered_gamma,
                                 epsilon=epsilon)
# ------------------------------------------------------------------------



# 0: ln-dot-dot, 1: dot-dot-ln
TEST_CASE = int(os.environ.get("TEST_CASE", 0))
assert TEST_CASE in (0, 1)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def layernorm(x, gamma, beta, zero_centered_gamma, epsilon):
    out = _layernorm(x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
    return out

def _layernorm(x, gamma, beta, zero_centered_gamma, epsilon):
  z, *_ = layernorm_fwd_p.bind(x, gamma, beta,
                               zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
  return z

def layernorm_fwd_rule(x, gamma, beta, zero_centered_gamma, epsilon):
  z, mu, rsigma = layernorm_fwd_p.bind(
      x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
  return z, (x, mu, rsigma, gamma)

def layernorm_bwd_rule(zero_centered_gamma, epsilon, res, dz):
  x, mu, rsigma, gamma = res
  dx, dgamma, dbeta = layernorm_bwd_p.bind(
      dz, x, mu, rsigma, gamma,
      zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
  return dx, dgamma, dbeta

layernorm.defvjp(layernorm_fwd_rule, layernorm_bwd_rule)


layernorm_fwd_p = core.Primitive('layernorm_fwd')
layernorm_fwd_p.multiple_results = True

@layernorm_fwd_p.def_abstract_eval
def _layernorm_fwd_abstract_eval(x_aval, gamma_aval, beta_aval, *,
                                 zero_centered_gamma, epsilon):
  # x_aval: [N, H]
  # gamma_aval: [H]
  # beta_aval: [H]
  del gamma_aval, beta_aval, epsilon
  out_aval = core.raise_to_shaped(x_aval)
  mu_aval = rsigma_aval = out_aval.update(shape=out_aval.shape[:-1])
  return out_aval, mu_aval, rsigma_aval

@layernorm_fwd_p.def_impl
def layernorm_fwd_impl(x, gamma, beta, zero_centered_gamma, epsilon):
  # FIXME (Ming Huang):
  # Expected
  # x_aval: [N, H]
  # gamma_aval: [H]
  # beta_aval: [H]
  assert gamma.ndim == beta.ndim == 1
  assert x.shape[-1] == gamma.shape[0] == beta.shape[0]

  normed, mu, rsigma = te_layernorm_fwd(
      x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
  return normed, mu, rsigma


# The vmap (batching) rule for layernorm_fwd_p just needs to put batch
# dimensions at the front.

from jax._src.interpreters import batching

def layernorm_fwd_batcher(
    batched_args: Sequence[jax.Array],
    batch_dims: Sequence[int | None],
    *,
    zero_centered_gamma: bool,
    epsilon: float,
) -> tuple[Sequence[jax.Array], Sequence[int | None]]:
  x, gamma, beta = batched_args
  x_bdim, gamma_bdim, beta_bdim = batch_dims
  out_bdims = x_bdim, gamma_bdim, beta_bdim
  # FIXME (Ming Huang):
  # Need a way to project x, gamma and beta on the batch_dim before calling layernorm_fwd_p.bind,
  # then layernorm_fwd_p does not need to know about batch info.
  # That is like named_axes of xmap.
  return layernorm_fwd_p.bind(x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon), out_bdims
batching.primitive_batchers[layernorm_fwd_p] = layernorm_fwd_batcher


from jax._src.interpreters import mlir
from jax.experimental.custom_partitioning import custom_partitioning

_layernorm_fwd_lower = custom_partitioning(layernorm_fwd_impl,
                                           static_argnums=(3, 4))

def infer_sharding_from_operands(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
  del epsilon, result_infos  # Unused.
  x_spec = get_padded_spec(arg_infos[0])
  # FIXME (Ming Huang): Why out_sharding is P(*x_spec[:-1]), instead of P(*x_spec)
  # I imagine the returns would be like (P(*x_spec), P(*x_spec[:-1]), P(*x_spec[:-1]))
  out_sharding = NamedSharding(mesh, P(*x_spec[:-1]))
  return (out_sharding,) * 3

def partition(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
  x_spec = get_padded_spec(arg_infos[0])
  arg_shardings = (NamedSharding(mesh, P(*x_spec[:-1], None)),
                   ) + (NamedSharding(mesh, P()),) * 2
  out_shardings = (NamedSharding(mesh, P(*x_spec[:-1])),) * 3
  impl = partial(layernorm_fwd_impl, zero_centered_gamma=zero_centered_gamma,
                 epsilon=epsilon)
  return mesh, impl, out_shardings, arg_shardings

_layernorm_fwd_lower.def_partition(
    infer_sharding_from_operands=infer_sharding_from_operands,
    partition=partition)

mlir.register_lowering(layernorm_fwd_p,
                       mlir.lower_fun(_layernorm_fwd_lower, multiple_results=True))



layernorm_bwd_p = core.Primitive('layernorm_bwd')
layernorm_bwd_p.multiple_results = True

@layernorm_bwd_p.def_abstract_eval
def layernorm_bwd_abstract_eval(dz_aval, _, __, ___, gamma_aval, *,
                                zero_centered_gamma, epsilon):
  dx_aval = core.raise_to_shaped(dz_aval)
  dgamma_aval = dbeta_aval = core.raise_to_shaped(gamma_aval)
  return dx_aval, dgamma_aval, dbeta_aval

@layernorm_bwd_p.def_impl
def layernorm_bwd_impl(dz, x, mu, rsigma, gamma, zero_centered_gamma, epsilon):
  pre = x.shape[:-1]
  dz = dz.reshape(-1, x.shape[-1])
  x = x.reshape(-1, x.shape[-1])
  mu = mu.reshape(-1)
  rsigma = rsigma.reshape(-1)
  # FIXME (Ming Huang):
  # Expected
  # dz_aval: [N, H]
  # x_aval: [N, H]
  # mu_aval: [N]
  # rsigma_aval: [N]
  # gamma_aval: [H]

  dx, dgamma, dbeta = te_layernorm_bwd(
      dz, mu, rsigma, x, gamma, zero_centered_gamma=zero_centered_gamma,
      epsilon=epsilon)
  return dx.reshape(*pre, -1), dgamma, dbeta

#Note that the batching rule for the bwd is not needed in this example,
#since we are doing grad-of-vmap, i.e. vmap would be in the network definition
def layernorm_bwd_batcher(
    batched_args: Sequence[jax.Array],
    batch_dims: Sequence[int | None],
    *,
    zero_centered_gamma: bool,
    epsilon: float,
) -> tuple[Sequence[jax.Array], Sequence[int | None]]:
  dz, x, mu, rsigma, gamma = batched_args
  _, x_bdim, _, _, gamma_bdim = batch_dims

  out_bdims = x_bdim, gamma_bdim, gamma_bdim
  # FIXME (Ming Huang):
  # Need a way to project x, gamma and beta on the batch_dim before calling layernorm_bwd_p.bind,
  # then layernorm_bwd_p does not need to know about batch info.
  # That is like named_axes of xmap.
  return layernorm_bwd_p.bind(dz, x, mu, rsigma, gamma, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon), out_bdims
batching.primitive_batchers[layernorm_bwd_p] = layernorm_bwd_batcher

_layernorm_bwd_lower = custom_partitioning(layernorm_bwd_impl, static_argnums=(5, 6))
def infer_sharding_from_operands(
    zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
  x_spec = get_padded_spec(arg_infos[0])
  dx_sharding = NamedSharding(mesh, P(*x_spec[:-1], None))
  dgamma_sharding = dbeta_sharding = NamedSharding(mesh, P())
  return dx_sharding, dgamma_sharding, dbeta_sharding

def partition(zero_centered_gamma, epsilon, mesh, arg_infos, result_infos):
  x_spec = get_padded_spec(arg_infos[1])
  dx_sharding = NamedSharding(mesh, P(*x_spec[:-1], None))
  dgamma_sharding = dbeta_sharding = NamedSharding(mesh, P())
  out_shardings = dx_sharding, dgamma_sharding, dbeta_sharding
  arg_shardings = (NamedSharding(mesh, P(*x_spec[:-1])),) * 4
  arg_shardings = (*arg_shardings, NamedSharding(mesh, P()))

  def sharded_impl(dz, x, mu, rsigma, gamma):
    local_dx, local_dgamma, local_dbeta = \
        layernorm_bwd_impl(
            dz, x, mu, rsigma, gamma, zero_centered_gamma=zero_centered_gamma,
            epsilon=epsilon)
    global_dgamma = jax.lax.psum(local_dgamma, filter_none(x_spec[:-1]))
    global_dbeta  = jax.lax.psum( local_dbeta, filter_none(x_spec[:-1]))
    return local_dx, global_dgamma, global_dbeta

  return mesh, sharded_impl, out_shardings, arg_shardings

def get_padded_spec(arg_info: jax.ShapeDtypeStruct) -> tuple:
  if arg_info.sharding is None:
    return (None,) * arg_info.ndim
  ndim, spec = arg_info.ndim, arg_info.sharding.spec
  assert len(spec) <= ndim
  return spec + (None,) * (ndim - len(spec))

def filter_none(xs: tuple) -> tuple:
  return tuple(x for x in xs if x is not None)

_layernorm_bwd_lower.def_partition(
    infer_sharding_from_operands=infer_sharding_from_operands,
    partition=partition
)


mlir.register_lowering(layernorm_bwd_p,
                       mlir.lower_fun(_layernorm_bwd_lower, multiple_results=True))

# Tests
def _layernorm(x, gamma, beta, layernorm_type, zero_centered_gamma, epsilon):
  assert layernorm_type is None
  del layernorm_type
  return layernorm(x, gamma, beta, zero_centered_gamma, epsilon)

def layernorm_ref(x, scale, bias, layernorm_type, zero_centered_gamma, epsilon):
    assert layernorm_type is None
    del layernorm_type

    x_ = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x_, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
    normed_input = (x_ - mean) * jax.lax.rsqrt(var + epsilon)
    if zero_centered_gamma:
        return jnp.asarray(normed_input * (scale + 1) + bias).astype(x.dtype)
    return jnp.asarray(normed_input * scale + bias).astype(x.dtype)

def func(x, gamma, beta, y1, y2, ln_fn):
    if TEST_CASE == 0:
        x = ln_fn(x, gamma, beta, None, False, 1e-6)
        x = jnp.dot(x, y1)
        out = jnp.dot(x, y2)
        return jnp.mean(out)
    else:
        x = jnp.dot(x, y1)
        x = jnp.dot(x, y2)
        out = ln_fn(x, gamma, beta, None, False, 1e-6)
        return jnp.mean(out)

def vmap_f(x, gamma, beta, y1, y2):
    partial_ln_func = partial(func, ln_fn=_layernorm)
    return jnp.mean(vmap(partial_ln_func, in_axes=(0, 0, 0, 0, 0))(x, gamma, beta, y1, y2))

def vmap_f_ref(x, gamma, beta, y1, y2):
    partial_ln_func = partial(func, ln_fn=layernorm_ref)
    return jnp.mean(vmap(partial_ln_func, in_axes=(0, 0, 0, 0, 0))(x, gamma, beta, y1, y2))


x_ = random.normal(random.PRNGKey(1124), (2, 32, 128))
gamma_ = jnp.ones((2, 128))
beta_ = jnp.ones((2, 128))
y1_ = random.normal(random.PRNGKey(1126), (2, 128, 128))
y2_ = random.normal(random.PRNGKey(1127), (2, 128, 128))

graded_f = jax.value_and_grad(vmap_f, argnums=(0, 1, 2, 3, 4))
graded_f_ref = jax.value_and_grad(vmap_f_ref, argnums=(0, 1, 2, 3, 4))

devices = np.array(jax.local_devices())
devices = devices.reshape((2, 2, 2))
with Mesh(devices, ('p', 'd', 't')) as mesh:
    x = jax.device_put(x_, NamedSharding(mesh, PartitionSpec('p', 'd', None)))
    gamma = jax.device_put(gamma_, NamedSharding(mesh, PartitionSpec('p', None)))
    beta = jax.device_put(beta_, NamedSharding(mesh, PartitionSpec('p', None)))
    y1 = jax.device_put(y1_, NamedSharding(mesh, PartitionSpec('p', None, 't')))
    y2 = jax.device_put(y2_, NamedSharding(mesh, PartitionSpec('p', 't', None)))

    pjitter_ref = pjit(graded_f_ref,
                    in_shardings=[PartitionSpec('p', 'd', None), PartitionSpec('p', None), PartitionSpec('p', None),
                                    PartitionSpec('p', None, 't'), PartitionSpec('p', 't', None)],
                    out_shardings=(None, (PartitionSpec('p', 'd', None), PartitionSpec('p', None), PartitionSpec('p', None),
                                            PartitionSpec('p', None, 't'), PartitionSpec('p', 't', None)))
                   )

    pjitter = pjit(graded_f,
                   in_shardings=[PartitionSpec('p', 'd', None), PartitionSpec('p', None), PartitionSpec('p', None),
                                 PartitionSpec('p', None, 't'), PartitionSpec('p', 't', None)],
                   out_shardings=(None, (PartitionSpec('p', 'd', None), PartitionSpec('p', None), PartitionSpec('p', None),
                                         PartitionSpec('p', None, 't'), PartitionSpec('p', 't', None)))
              )

    ref_l, ref_grads = pjitter_ref(x_, gamma_, beta_, y1_, y2_)
    test_l, test_grads = pjitter(x_, gamma_, beta_, y1_, y2_)

print("TEST_CASE:", TEST_CASE)
print("loss match:", jnp.allclose(ref_l, test_l))
print("dgrad match:", jnp.allclose(ref_grads[0], test_grads[0], rtol=1e-04))
print("dgamma match:", jnp.allclose(ref_grads[1], test_grads[1]))
print("dbeta match:", jnp.allclose(ref_grads[2], test_grads[2], rtol=1e-03))
print("dy1 match:", jnp.allclose(ref_grads[3], test_grads[3]))
print("dy2 match:", jnp.allclose(ref_grads[4], test_grads[4]))
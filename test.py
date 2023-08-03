import os
os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_proto --xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_to=custom_part_ln_dump"

from functools import partial

import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental.custom_partitioning import custom_partitioning
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec, NamedSharding

from transformer_engine.jax.cpp_extensions import layernorm_fwd, layernorm_bwd

# 0: ln-dot-dot, 1: dot-dot-ln
TEST_CASE = int(os.environ.get("TEST_CASE", 0))
assert TEST_CASE in (0, 1)

@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _layernorm(x, gamma, beta, layernorm_type, zero_centered_gamma, epsilon):
    outputs = _layernorm_fwd_custom_p(x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
    return outputs


@partial(custom_partitioning, static_argnums=(3, 4))
def _layernorm_fwd_custom_p(x, gamma, beta, zero_centered_gamma, epsilon):
    outputs = layernorm_fwd(x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
    return outputs

@partial(custom_partitioning, static_argnums=(5, 6))
def _layernorm_bwd_custom_p(g, mu, rsigma, x, gamma, zero_centered_gamma, epsilon):
    outputs = layernorm_bwd(g, mu, rsigma, x, gamma,
                            zero_centered_gamma=zero_centered_gamma,
                            epsilon=epsilon)
    return outputs

def _layernorm_fwd(x, gamma, beta, layernorm_type, zero_centered_gamma, epsilon):
    outputs = _layernorm(x, gamma, beta, layernorm_type, zero_centered_gamma, epsilon)
    _, mu, rsigma = outputs
    return outputs, (mu, rsigma, x, gamma)

def _layernorm_bwd(layernorm_type, zero_centered_gamma, epsilon, ctx, g):
    mu, rsigma, x, gamma = ctx

    grad_input, grad_gamma, grad_beta = _layernorm_bwd_custom_p(g[0], mu, rsigma, x, gamma,
                                                      zero_centered_gamma=zero_centered_gamma,
                                                      epsilon=epsilon)
    return grad_input, grad_gamma, grad_beta

_layernorm.defvjp(_layernorm_fwd, _layernorm_bwd)


def force_not_sharded_dim(shard, dim):
    # Utility to force a dim to NOT be sharded
    # TODO: For updated JAX, convert shard to a positional sharding to be sure to cover 100% cases.
    assert isinstance(shard, NamedSharding)
    if len(shard.spec) <= dim:
        # Replicated as not specified.
        return shard
    if shard.spec[dim] is None:
        # Specified as Sharded
        return shard
    return NamedSharding(shard.mesh, PartitionSpec(*shard.spec[:dim], None, *shard.spec[dim+1:]))

def partition_ln_fwd(zero_centered_gamma, epsilon, arg_infos, result_infos):
    def _impl(x, gamma, beta):
        outputs = layernorm_fwd(x, gamma, beta, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
        return outputs

    arg_shard = force_not_sharded_dim(arg_infos[0].sharding, 1), \
                force_not_sharded_dim(arg_infos[1].sharding, 0), \
                force_not_sharded_dim(arg_infos[2].sharding, 0)
    # The list and tuple bellow are important. If you change those type, you will get errors.
    return _impl, [a.sharding for a in result_infos], arg_shard

def infer_sharding_from_operands_ln_fwd(zero_centered_gamma, epsilon, arg_infos, out_shape):
    rsigma_sharding = NamedSharding(mesh, PartitionSpec(arg_infos[0].sharding.spec[0]))
    # Must return a list, not a tuple.
    return [force_not_sharded_dim(arg_infos[0].sharding, 1), rsigma_sharding, rsigma_sharding]

_layernorm_fwd_custom_p.def_partition(
    infer_sharding_from_operands=infer_sharding_from_operands_ln_fwd,
    partition=partition_ln_fwd)

def partition_ln_bwd(zero_centered_gamma, epsilon, arg_infos, result_infos):
    def _impl(g, mu, rsigma, x, gamma):
        outputs = layernorm_bwd(g, mu, rsigma, x, gamma, zero_centered_gamma=zero_centered_gamma, epsilon=epsilon)
        return outputs

    arg_shard = (
        arg_infos[0].sharding,
        arg_infos[1].sharding,
        arg_infos[2].sharding,
        arg_infos[3].sharding,
        arg_infos[4].sharding
    )
    # The list and tuple bellow are important. If you change those type, you will get errors.
    return _impl, [a.sharding for a in result_infos], arg_shard

def infer_sharding_from_operands_ln_bwd(zero_centered_gamma, epsilon, arg_infos, out_shape):
    # TODO (Ming Huang): Backward cannot automatically trigger all-reduce along mesh's x-axis for dgamma and dbeta.
    wgrads = NamedSharding(mesh, PartitionSpec(None))
    # TODO (Ming Huang): Why grad_output have no sharding info? It should be sharded along x-axis in the mesh.
    # It could be sharding=None, or Sharding=PartitionSpec(), but expect to get PartitionSpec('x')
    assert arg_infos[0].sharding is not None
    return [force_not_sharded_dim(arg_infos[0].sharding, 1), wgrads, wgrads]



_layernorm_bwd_custom_p.def_partition(
    infer_sharding_from_operands=infer_sharding_from_operands_ln_bwd,
    partition=partition_ln_bwd)

def func(x, gamma, beta, y1, y2):
    if TEST_CASE == 0:
        x, _, _ = _layernorm(x, gamma, beta, None, False, 1e-6)
        x = jnp.dot(x, y1)
        out = jnp.dot(x, y2)
        return jnp.mean(out)
    else:
        x = jnp.dot(x, y1)
        x = jnp.dot(x, y2)
        out, _, _ = _layernorm(x, gamma, beta, None, False, 1e-6)
        return jnp.mean(out)


x_ = random.normal(random.PRNGKey(1124), (32, 128))
gamma_ = jnp.ones((128,))
beta_ = jnp.ones((128,))
y1_ = random.normal(random.PRNGKey(1126), (128, 128))
y2_ = random.normal(random.PRNGKey(1127), (128, 128))

graded_f = jax.value_and_grad(func, argnums=(0, 1, 2, 3, 4))
ref_l, ref_grads = graded_f(x_, gamma_, beta_, y1_, y2_)

devices = np.array(jax.local_devices()).reshape((4, 2))
with Mesh(devices, ('x', 'y')) as mesh:
    x = jax.device_put(x_, NamedSharding(mesh, PartitionSpec('x', None)))
    gamma = jax.device_put(gamma_, NamedSharding(mesh, PartitionSpec(None)))
    beta = jax.device_put(beta_, NamedSharding(mesh, PartitionSpec(None)))
    y1 = jax.device_put(y1_, NamedSharding(mesh, PartitionSpec(None, 'y')))
    y2 = jax.device_put(y2_, NamedSharding(mesh, PartitionSpec('y', None)))

    pjitter = pjit(graded_f,
                   in_shardings=[PartitionSpec('x', None), PartitionSpec(None), PartitionSpec(None),
                                      PartitionSpec(None, 'y'), PartitionSpec('y', None)],
                   out_shardings=(None, (PartitionSpec('x', None), PartitionSpec(None), PartitionSpec(None),
                                              PartitionSpec(None, 'y'), PartitionSpec('y', None)))
              )

    test_l, test_grads = pjitter(x, gamma, beta, y1, y2)

print("TEST_CASE:", TEST_CASE)
print("loss match:", jnp.allclose(ref_l, test_l))
print("dgrad match:", jnp.allclose(ref_grads[0], test_grads[0]))
print("dgamma match:", jnp.allclose(ref_grads[1], test_grads[1]))
print("dbeta match:", jnp.allclose(ref_grads[2], test_grads[2]))
print("dy1 match:", jnp.allclose(ref_grads[3], test_grads[3]))
print("dy2 match:", jnp.allclose(ref_grads[4], test_grads[4]))

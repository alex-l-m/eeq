import torch as t
from torch_geometric.utils import to_dense_batch, to_dense_adj
import einops as ein

def constrained_newton_update(constraint_lhs, gradient, hessian):
    '''Minimize a quadratic function on a subspace. The subspace is taken to be
    all v such that constraint_lhs v = 0. The quadratic function is gradient +
    v^T hessian v. All inputs are assumed to be batch, with the batch dimension
    as the first dimension.

    constraint_lhs: batch_size x n_cons x dim tensor
    gradient: batch_size x dim tensor
    hessian: batch_size x dim x dim tensor

    If using the gradient and hessian of a function f, iteratively applying this
    function implements Newton's algorithm on f'''

    # Should be a square with batch dimension 0
    batch_size = hessian.shape[0]
    dim = hessian.shape[1]
    assert hessian.shape[2] == dim
    # Number of constraint equations
    # Assumes batch dimension is 0
    assert constraint_lhs.shape[0] == batch_size
    n_cons = constraint_lhs.shape[1]
    assert constraint_lhs.shape[2] == dim
    # Gradient also has to have the same batch size, and dim rows
    assert gradient.shape[0] == batch_size
    assert gradient.shape[1] == dim
    # Basis for the orthogonal complement of the subspace of allowed solutions
    orth_comp_basis = constraint_lhs.transpose(2, 1)
    lhs = \
            t.cat([
                t.cat([-1 * hessian, orth_comp_basis], dim = 2),
                t.cat([constraint_lhs, t.zeros((batch_size, n_cons, n_cons), dtype = t.float)], dim = 2)
            ], dim = 1)
    rhs = t.cat([gradient, t.zeros((batch_size, n_cons), dtype = t.float)], dim = 1)
    solution_with_lagrange_multiplier = t.linalg.solve(lhs, rhs)
    # Include all batches, but only the part of the solution without the
    # lagrange multiplier
    solution = solution_with_lagrange_multiplier[:,:dim]
    return solution
# WolframAlpha: coulomb's constant in eV * angstrom / (electron charge)^2
# https://www.wolframalpha.com/input?i=coulomb%27s+constant+in+eV+*+angstrom+%2F+%28electron+charge%29%5E2
coulomb = 14.39964548

def eeq(batch, m_a, h_a):
    '''Electronegativity equalization for PyTorch Geometric.

    batch: PyTorch Geometric batch
    m_a: Chemical potential of each atom, as a one-dimensional tensor
    h_a: Hardness of each atom, as a one-dimensional tensor

    It is assumed that all pairs of interacting atoms are connected by an edge,
    and there is only one edge attribute, the distance between the atoms.'''
    # All the matrices I'm going to need, in the correct shape
    # Extend the linear systems to include empty spots
    # This goes on the right-hand side, so we want to fill with 0's
    # That way, the charge of the empty spots will be set to zero
    mu, not_fake = to_dense_batch(m_a, batch.batch,
                                  fill_value = 0)
    # Get max size, used when creating the off-diagonal elements of the
    # hardness kernel
    max_size = mu.shape[1]
    # Get number of atoms in each molecule, used for generating constraints
    n_atoms = ein.reduce(not_fake, "batch atom -> batch", "sum")
    # The diagonal elements: hardness
    # Hardness of atoms (rather than elements)
    # Extend to include empty spots
    # This goes on the left-hand side, so we want to fill with 1's
    # That way, we get the equation:
    # 1 * charge of empty spot = 0
    batched_h, not_fake = to_dense_batch(h_a, batch.batch,
                                        fill_value = 1)
    diagonal = t.diag_embed(batched_h)

    # The off-diagonal elements: Coulomb interactions
    # Make a vector of distances from the matrix of features
    distance_vector = ein.rearrange(batch.edge_attr, "atom () -> atom")
    interaction_vector = coulomb / distance_vector
    off_diagonal = to_dense_adj(batch.edge_index, batch.batch,
                                interaction_vector, max_size)

    H = diagonal + off_diagonal

    # A row vector, since my code handles more than one constraint
    # Awkward way to get batch size
    batch_size = mu.shape[0]
    # Only the real atoms are constrained to sum to 1
    constraint = ein.rearrange(
            [t.cat([t.ones(n, dtype = t.float),
                    t.zeros(max_size - n, dtype = t.float)]) \
            for n in list(n_atoms)],
            "batch atom -> batch 1 atom")

    # "Excess" density (electron population minus nuclear charge), including
    # 0 density at the empty positions
    excess_density = constrained_newton_update(constraint, mu, H)
    # Excess density is effectively negative electric charge, convert to charge
    charge = -1 * excess_density

    # Linearize and filter the batched charge
    charge_linear = ein.rearrange(charge, "batch atom -> (batch atom)")
    not_fake_linear = ein.rearrange(not_fake, "batch atom -> (batch atom)")
    charge_filtered = charge_linear[not_fake_linear]
    return charge_filtered

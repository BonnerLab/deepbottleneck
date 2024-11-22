import numpy as np
import torch

def compute_johnson_lindenstrauss_limit(*, n_samples: int, epsilon: float) -> int:
    return int(np.ceil(4 * np.log(n_samples) / ((epsilon**2) / 2 - (epsilon**3) / 3)))

def create_sparse_projection_matrix(
    *,
    n_features: int,
    n_components: int,
    density: float | None = None,
    seed: int = 0,
) -> torch.Tensor:
    assert isinstance(n_features, int), "n_features must be an int"
    assert n_features > 1, "n_features must be > 1"

    if density is None:
        density = np.exp(-np.log(n_features) / 2)
    else:
        assert isinstance(density, float)
        assert density > 0, "density must be > 0"
        assert density <= 1, "density must be <= 1"

    assert isinstance(n_components, int), "n_components must be an int"
    assert n_components >= 1, "n_components must be >= 1"

    scale = np.exp(-(np.log(density) + np.log(n_components)) / 2)

    n_elements = n_features * n_components

    rng = np.random.default_rng(seed=seed)
    n_nonzero = rng.binomial(n=n_elements, p=density, size=1)[0]
    indices = rng.choice(a=n_elements, size=n_nonzero, replace=False).astype(np.int64)
    locations = np.stack(
        np.unravel_index(indices=indices, shape=(n_features, n_components)),
    )

    projection = torch.sparse_coo_tensor(
        indices=torch.from_numpy(locations),
        values=scale
        * (2 * rng.binomial(n=1, p=0.5, size=n_nonzero) - 1).astype(np.float32),
        size=(n_features, n_components),
    )
    return projection

class SparseRandomProjection:
    def __init__(
        self,
        *,
        n_components: int,
        density: float | None = None,
        seed: int = 0,
        allow_expansion: bool = False,
    ) -> None:
        self.n_components = n_components
        self.density = density
        self.seed = seed
        self.allow_expansion = allow_expansion
        self.projection = None  # Cache for the projection matrix

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        features = features.flatten(start_dim=1)
        n_features = features.shape[-1]

        # Check if we have already created the projection matrix
        if self.projection is None or self.projection.shape[0] != n_features:
            self.projection = create_sparse_projection_matrix(
                n_features=n_features,
                n_components=self.n_components,
                density=self.density,
                seed=self.seed,
            )

        if (n_features <= self.projection.shape[-1]) and not self.allow_expansion:
            return features

        return self._project(features=features, projection=self.projection)

    def _project(
        self,
        *,
        features: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(features, projection.to_dense())

def batched_srp(xtrain, xtest, epsilon=0.1, batch_size=10):
    
    # 1. Compute n_components
    n_samples = xtrain.shape[0] + xtest.shape[0]
    n_components = compute_johnson_lindenstrauss_limit(n_samples=n_samples, epsilon=epsilon)
    
    # 2. Initialise sparse_random_projection
    sparse_random_projection = SparseRandomProjection(
        n_components=n_components,
        density=None,
        seed=0,
        allow_expansion=False
    )
    # sparse_random_projection = sparse_random_projection.cuda()
    
    # 3. Initialise empty list for xtrain_reduced
    xtrain_reduced = []

    # 4. Process xtrain in batches
    for i in range(0, xtrain.shape[0], batch_size):
        batch_input = xtrain[i:i + batch_size]
        output = sparse_random_projection(batch_input)
        xtrain_reduced.append(output)
    
    # 5. Concatenate xtrain_reduced into a torch tensor
    xtrain_reduced = torch.cat(xtrain_reduced, dim=0)
    
    # 6. Initialise empty list for xtest_reduced and process xtest in batches
    xtest_reduced = []
    for i in range(0, xtest.shape[0], batch_size):
        batch_input = xtest[i:i + batch_size]
        output = sparse_random_projection(batch_input)
        xtest_reduced.append(output)
    
    # Concatenate xtest_reduced into a torch tensor
    xtest_reduced = torch.cat(xtest_reduced, dim=0)
    
    # 7. Return xtrain_reduced and xtest_reduced
    return xtrain_reduced, xtest_reduced
from main import *

def load_model_modified(num_channels):
    arch = ["Fw", "Std", "P", "N"]
    psi_arch = ["mod"]
    num_channels_in = 3
    spatial_size = 224
    num_blocks = len(num_channels)

    input_type = TensorType(
        num_channels=num_channels_in,
        spatial_shape=(spatial_size, spatial_size),
        complex=False,
    )

    builder = Builder(input_type)

    builder.add_layer(ToSplitTensor, dict(groups={(0): num_channels_in}))
    for i in range(num_blocks):
        builder.add_layer(
            build_layers,
            dict(modules=arch, psi_modules=psi_arch, i=i, num_channels=num_channels),
        )

    builder.add_layer(ToTensor)
    model = builder.module()

    return model

def init_conv(channel_sizes):
    nchannels = len(channel_sizes)
    conv_layers = nn.ModuleList()

    for i in range(1, nchannels + 1):
        # Projection matrix
        file_path_pm = f"/data/apassi1/variance-coefs/pm_cent{i}.npy"
        projection_matrix = np.load(file_path_pm)
        projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
        projection_matrix = projection_matrix.unsqueeze(-1).unsqueeze(-1)

        # Bias
        file_path_bias = f"/data/apassi1/variance-coefs/bias{i}.npy"
        bias = np.load(file_path_bias)
        bias = torch.from_numpy(bias)

        p = projection_matrix[: channel_sizes[i - 1], :, :, :]
        b = bias[: channel_sizes[i - 1]]
                
        # Initializing layer with a unique name
        conv_layer = nn.Conv2d(
            projection_matrix.shape[0],
            out_channels=channel_sizes[i - 1],
            kernel_size=1,
            bias=True,
        )
        conv_layer.bias = nn.Parameter(b)
        conv_layer.weight = nn.Parameter(p)
        conv_layers.append(conv_layer)

    return conv_layers

def init_conv_rand(channel_sizes):
    nchannels = len(channel_sizes)
    conv_layers = nn.ModuleList()

    for i in range(1, nchannels + 1):
        
        if i == 1: 
            input_channels = 27
        else:
             input_channels = channel_sizes[i-2]*9

        conv_layer = nn.Conv2d(
            in_channels=input_channels,
            out_channels=channel_sizes[i - 1],
            kernel_size=1,
            bias=True,
        )
        conv_layers.append(conv_layer)

    return conv_layers

def init_splitter(channel_sizes):
    nchannels = len(channel_sizes)
    splitter_layers = nn.ModuleList()

    for i in range(1, nchannels + 1):
        input_type = TensorType(
            num_channels=channel_sizes[i - 1], spatial_shape=(112, 112), complex=False
        )
        groups = {0: channel_sizes[i - 1]}
        layer = ToSplitTensor(input_type=input_type, groups=groups)

        splitter_layers.append(layer)

    return splitter_layers


def make_pca(channel_sizes):
    # loading model
    m = load_model_new(channel_sizes)
    nchannels = len(channel_sizes)

    # selecting useful layers in the model
    totensor = m[-1]
    norm = m[1][4]

    # initialising 1x1conv and splitter layers
    conv_layers = init_conv(channel_sizes)
    splitter_layers = init_splitter(channel_sizes)

    # initialising new model
    mnew = nn.Sequential(
        m[0],
    )

    for i in range(1, nchannels + 1):
        mnew.append(m[i][:3])
        mnew[i].add_module("3", totensor)
        mnew[i].add_module("4", conv_layers[i - 1])
        mnew[i].add_module("5", splitter_layers[i - 1])
        mnew[i].add_module("6", norm)

    mnew.append(totensor)
    return mnew


def make_random(channel_sizes):
    # loading model
    m = load_model_modified(channel_sizes)
    nchannels = len(channel_sizes)

    # selecting useful layers in the model
    totensor = m[-1]
    norm = m[1][4]

    # initialising 1x1conv and splitter layers
    conv_layers = init_conv_rand(channel_sizes)
    splitter_layers = init_splitter(channel_sizes)

    # initialising new model
    mnew = nn.Sequential(
        m[0],
    )

    for i in range(1, nchannels + 1):
        mnew.append(m[i][:3])
        mnew[i].add_module("3", totensor)
        mnew[i].add_module("4", conv_layers[i - 1])
        mnew[i].add_module("5", splitter_layers[i - 1])
        mnew[i].add_module("6", norm)

    mnew.append(totensor)
    return mnew
        
class New_Standardization(nn.Module):
    def __init__(self, input_type: TensorType, layer_number: int, dim=1, shape=None, remove_mean=True, eps=1e-05, momentum=0.1):
        super().__init__()
        self.input_type = input_type
        self.layer_number = layer_number
        self.dim = to_tuple(dim)  # Non-negative and strictly increasing
        assert self.dim[0] >= 0 and all(self.dim[i + 1] > self.dim[i] for i in range(len(self.dim) - 1))
        if shape is None:
            assert self.dim == (1,)
            self.shape = (self.input_type.num_channels,)
        else:
            self.shape = to_tuple(shape)  # Same length as self.dim
            assert len(self.shape) == len(self.dim)
        self.complex = self.input_type.complex
        self.output_type = self.input_type
        self.remove_mean = remove_mean

        mean_path = f"/data/apassi1/std_coefs/mean{layer_number}.npy"
        var_path = f"/data/apassi1/std_coefs/var{layer_number}.npy" 
        mean = torch.tensor(np.load(mean_path), dtype=torch.float32)
        var = torch.tensor(np.load(var_path), dtype=torch.float32)
        self.register_buffer("mean", mean)  # complex or real
        self.register_buffer("var", var)  # real
        self.eps = eps
        self.momentum = momentum

    def extra_repr(self) -> str:
        return f"dim={self.dim}, shape={self.shape}, complex={self.complex}, remove_mean={self.remove_mean}, layer_number={self.layer_number}"

    def forward(self, x: Tensor) -> Tensor:
        index = tuple(Ellipsis if i in self.dim else None for i in range(x.ndim))
        x = (x - self.mean[index]) * torch.rsqrt(self.eps + self.var[index])
        return x
    
def build_layers_new(
    input_type: SplitTensorType,
    modules,
    psi_modules=[],
    i: int = 0,
    num_channels: List[int] = [],
    layer_number: int = 0
):
    """Builds a list of layers from a list of modules.
    :param input_type:
    :param modules: list of strings describing the layers
    :poaram psi_modules: list of strings describing the layers applied after the high-frequency filters
    :param i: index of the current block
    :param num_channels: number of output channels for each block
    :return: Sequential module
    """
    builder = Builder(input_type)

    def branching_kwargs(**submodules):
        """Builds kwargs for Branching. Expects a dict of module_name -> architecture list of strings."""
        kwargs = {}
        for name, arch in submodules.items():
            kwargs[f"{name}_module_class"] = build_layers
            kwargs[f"{name}_module_kwargs"] = dict(modules=arch, i=i)
        return kwargs

    for module in modules:
        if module == "Fw":
            kwargs = dict(
                scales_per_octave=1,
                L=8,
                full_angles=False,
            )
            two_blocks_per_scale_after_block = len(num_channels) % 2
            if i >= two_blocks_per_scale_after_block:
                kwargs.update(
                    factorize_filters=True, i=(i - two_blocks_per_scale_after_block) % 2
                )

            builder.add_layer(Scattering2D, kwargs)

            kwargs = dict(phi=[], psi=psi_modules)
            builder.add_layer(Branching, branching_kwargs(**kwargs))

        elif module == "R":
            builder.add_batched(Realifier)

        elif module == "C":
            builder.add_batched(Complexifier)

        elif module in ["mod", "rho"]:
            kwargs = dict(
                non_linearity="mod",
                bias=None,
                gain=None,
                learned_params=False,
            )
            if module == "mod":
                builder.add_layer(ScatNonLinearity, kwargs)
            elif module == "rho":
                builder.add_layer(ScatNonLinearityAndSkip, kwargs)
                builder.add_layer(
                    Branching, branching_kwargs(linear=[], non_linear=[])
                )  # Possibility to add different modules to the linear/non_linear part

        elif module == "Std":
            builder.add_batched(New_Standardization, dict(layer_number=layer_number, remove_mean=True))


        elif module in ["P", "Pr", "Pc"]:
            out_channels = {0: num_channels[i]}
            # Determine type of weights (default is type of input).
            complex_weights = dict(P=None, Pr=True, Pc=False)[module]

            kwargs = dict(
                complex_weights=complex_weights,
                out_channels=out_channels,
            )

            builder.add_diagonal(ComplexConv2d, kwargs)

        elif module == "N":
            builder.add_diagonal(Normalization)

        elif module == "id":
            builder.add_layer(Identity)

        else:
            assert False

    return builder.module()

def load_model_new(num_channels):
    
    arch = ["Fw", "Std", "P", "N"]
    psi_arch = ["mod"]
    num_channels_in = 3
    spatial_size = 224
    num_blocks = len(num_channels)

    input_type = TensorType(
        num_channels=num_channels_in,
        spatial_shape=(spatial_size, spatial_size),
        complex=False,
    )
    builder = Builder(input_type)

    builder.add_layer(ToSplitTensor, dict(groups={(0): num_channels_in}))
    
    lid = 1
    for i in range(num_blocks):
        builder.add_layer(
            build_layers_new,
            dict(modules=arch, psi_modules=psi_arch, i=i, num_channels=num_channels, layer_number = lid),
        )
        lid += 1
        
    builder.add_layer(ToTensor)
    model = builder.module()
    return model
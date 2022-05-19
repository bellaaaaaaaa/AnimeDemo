def load_face_morpher(file_name: str):
    fma = FaceMorpher08Args()
    ba = BlockArgs()
    ba.set_initialization_method='he'
    ba.set_use_spectral_norm=False
    ba.set_normalization_layer_factory=InstanceNorm2dFactory()
    ba.set_nonlinearity_factory=ReLUFactory(inplace=False)

    fma.block_args = ba
    fma.set_image_size = 192
    fma.set_image_channels=4
    fma.set_num_expression_params=27
    fma.set_start_channels=64
    fma.set_bottleneck_image_size=24
    fma.set_num_bottleneck_blocks=6
    fma.set_max_channels=512

    factory = FaceMorpher08Factory(fma)
    #print("Loading the face morpher ... ", end="")
    module = factory.create()
    module.load_state_dict(torch_load(file_name))
    #print("DONE")
    return module
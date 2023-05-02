def define_model(configs, gpu):
    print(configs)
    model = None
    model_name = configs.name.lower()

    if model_name == "scunet":
        from archs.SCUNet.models import Generator

        model = Generator(configs).to(gpu)

    elif model_name == "realesrgan":
        from archs.RealESRGAN.models import Generator

        model = Generator(configs).to(gpu)

    elif model_name == "bsrgan":
        from archs.BSRGAN.models import Generator

        model = Generator(configs).to(gpu)

    elif model_name == "edsr":
        from archs.EDSR.models import Generator

        model = Generator(configs).to(gpu)

    elif model_name == "swinir":
        from archs.SwinIR.models import Generator

        model = Generator(configs).to(gpu)
    elif model_name == "ecbsr":
        from archs.ECBSR.models import Generator

        model = Generator(configs).to(gpu)
    elif model_name == "rlfn":
        from archs.RLFN.models import Generator

        model = Generator(configs).to(gpu)

    elif model_name == "rrsr_train":
        from archs.RRSR.train_models import Generator

        model = Generator(configs).to(gpu)
    elif model_name == "rrsr_infer":
        from archs.RRSR.infer_models import Generator

        model = Generator(configs).to(gpu)
    elif model_name == "unet_discriminator":
        from archs.Discriminators.unet_discriminator import Discriminator

        model = Discriminator(configs).to(gpu)

    elif model_name == "patchgan_discriminator":
        from archs.Discriminators.patchgan_discriminator import Discriminator

        model = Discriminator(configs).to(gpu)

    assert model != None, "Model is None"

    print(f"Architecture: {model_name} is going to be used")
    return model

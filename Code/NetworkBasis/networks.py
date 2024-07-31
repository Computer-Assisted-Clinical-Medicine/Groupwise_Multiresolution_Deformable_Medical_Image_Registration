import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI
from tensorflow.keras import layers
import tensorflow as tf
import voxelmorph as vxm
from voxelmorph import layers
from voxelmorph.tf.modelio import LoadableModel, store_config_args
import neurite as ne

from NetworkBasis import config as cfg

# make ModelCheckpointParallel directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel

class Network(LoadableModel):
    """
    Neural Network for groupwise registration between multiple images.
    Multiresolution: 4 Steps/Resolutions, 1/8 1/4 1/2 1, https://github.com/Computer-Assisted-Clinical-Medicine/Multistep_Networks_for_Deformable_Medical_Image_Registration
    """

    @store_config_args
    def __init__(self,inshape):
        """
        Parameters:
            inshape: Input shape. e.g. (192,192,64)
        """
        # configure multiresolution network
        src_feats = cfg.nb
        nb_features = [[16, 32, 32], [32, 32, 32, 32, 16, 16]]
        down_factor=[8,4,2,1]
        nb_steps=4

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')

        input_model = tf.keras.Model(inputs=source, outputs=source)

        moved_images=[None]*cfg.nb
        disp_sums=[None]*cfg.nb

        source_down=ne.layers.Resize((1/down_factor[0]))(source)

        output = VxmDense_adapted(inshape=source_down.shape[1:-1], nb_unet_features=nb_features, int_downsize=1, src_feats=src_feats)([source_down])

        for i in range(cfg.nb):
            disp_sums[i]=tf.cast(layers.RescaleTransform(down_factor[0])(output[:, :, :, :, 3*i:3*i+3]), 'float32')

        spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')


        for i in range(cfg.nb):
            moved_images[i] = spatial_transformer([source[:, :, :, :, i:i+1], disp_sums[i]])

        moved_image = tf.concat(moved_images, -1)
        moved_image_down = ne.layers.Resize((1 / down_factor[1]))(moved_image)

        output = VxmDense_adapted(inshape=moved_image_down.shape[1:-1], nb_unet_features=nb_features, int_downsize=1, src_feats=src_feats, name = '1')(moved_image_down)

        for i in range(cfg.nb):
            disp_sums[i] = disp_sums[i] + tf.cast(layers.RescaleTransform(down_factor[1])(output[:, :, :, :, 3*i:3*i+3]), 'float32')

        for i in range(cfg.nb):
            moved_images[i] = spatial_transformer([source[:, :, :, :, i:i + 1], disp_sums[i]])

        moved_image = tf.concat(moved_images, -1)
        moved_image_down = ne.layers.Resize((1 / down_factor[2]))(moved_image)

        output = VxmDense_adapted(inshape=moved_image_down.shape[1:-1], nb_unet_features=nb_features, int_downsize=1, src_feats=src_feats, name = '2')(moved_image_down)

        for i in range(cfg.nb):
            disp_sums[i] = disp_sums[i] + tf.cast(layers.RescaleTransform(down_factor[2])(output[:, :, :, :, 3 * i:3 * i + 3]), 'float32')

        for i in range(cfg.nb):
            moved_images[i] = spatial_transformer([source[:, :, :, :, i:i + 1], disp_sums[i]])

        moved_image = tf.concat(moved_images, -1)

        output = VxmDense_adapted(inshape=moved_image.shape[1:-1], nb_unet_features=nb_features, int_downsize=1, src_feats=src_feats, name = '3')(moved_image)
        for i in range(cfg.nb):
            disp_sums[i] = disp_sums[i] + tf.cast(layers.RescaleTransform(down_factor[3])(output[:, :, :, :, 3 * i:3 * i + 3]), 'float32')

        for i in range(cfg.nb):
            moved_images[i] = spatial_transformer([source[:, :, :, :, i:i + 1], disp_sums[i]])

        outputs = []
        for i in range(cfg.nb):
            outputs.append(moved_images[i])
            outputs.append(disp_sums[i])

        super().__init__(name='GroupwiseMultiresolutionNetwork', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.y_source = moved_images
        self.references.pos_flow = disp_sums

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

class VxmDense_adapted(LoadableModel):
    """
    Adapted from: https://github.com/voxelmorph/voxelmorph
    VoxelMorph network for (unsupervised) nonlinear registration between >2 images.
    """

    @store_config_args
    def __init__(self,
            inshape,
            nb_unet_features=None,
            nb_unet_levels=None,
            unet_feat_mult=1,
            nb_unet_conv_per_level=1,
            int_steps=0,
            int_downsize=2,
            bidir=False,
            use_probs=False,
            src_feats=1,
            trg_feats=1,
            unet_half_res=False,
            input_model=None,
            name=""):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. Default is False.
            input_model: Model to replace default input layer before concatenation. Default is None.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3, 4], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
            input_model = tf.keras.Model(inputs=source, outputs=source)
        else:
            source = input_model.outputs

        # build core unet model and grab inputs
        unet_model = vxm.networks.Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res
        )

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        nb_features_flow=ndims*src_feats #flow fields for all source images are stacked in flow_mean
        flow_mean = Conv(nb_features_flow, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='flow')(unet_model.output)

        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=KI.Constant(value=-10),
                            name='log_sigma')(unet_model.output)
            flow = ne.layers.SampleNormalLogVar(name="z_sample")([flow_mean, flow_logsigma])
        else:
            flow = flow_mean

        if not unet_half_res:
            # optionally resize for integration
            if int_steps > 0 and int_downsize > 1:
                flow = layers.RescaleTransform(1 / int_downsize, name='flow_resize')(flow)


        # optionally negate flow for bidirectional model
        pos_flow = flow
        if bidir:
            neg_flow = ne.layers.Negate(name='neg_flow')(flow)

        outputs = pos_flow
        super().__init__(name='vxm_dense'+name, inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if bidir else None

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])
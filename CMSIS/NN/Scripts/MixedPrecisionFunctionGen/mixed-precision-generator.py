import errno
import os
from mako.template import Template


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


CMSISInstallPath = cwd = os.getcwd() + "/../../"
CMSISSrcDirs = {'Include': CMSISInstallPath + "Include",
                'convolution': CMSISInstallPath + "Source/ConvolutionFunctions/",
                'fullyConnected': CMSISInstallPath + "Source/FullyConnectedFunctions/",
                'Pooling': CMSISInstallPath + "Source/PoolingFunctions/",
                'NNSupport': CMSISInstallPath + "Source/NNSupportFunctions/"}
CMSISDataPrecisions = ['u8', 'u4', 'u2']
CMSISQuantizationMethods = ['PACT', 'PACT_CH']
CMSISFoldingMethods = ['weights', 'icn'] 
CMSISConstrains = {'u8': 4,
                   'u4': 8,
                   'u2': 16}
CMSISAPI = "\n"
CMSISSupportAPI = "\n"


class CMSISFactory(object):
    def __init__(self, in_data_t, out_data_t, wt_data_t):
        self.in_data_t = in_data_t
        self.out_data_t = out_data_t
        self.wt_data_t = wt_data_t
        self.arithmetic_t = 'int16'
        self.fn_name = ''
        self.filename = ''
        self.quantization = ''
        self.folding = ''
        self.api = ''

    def generate_api(self):
        return Template(filename="templates/arm_nn_api.h").render(config=self)


class CMSISConvolve(CMSISFactory):
    def __init__(self, in_data_t, out_data_t, wt_data_t, quantization, folding):
        super().__init__(in_data_t, out_data_t, wt_data_t)
        self.fn_name = "arm_convolve_HWC_{0}_{1}_{2}{3}{4}".format(str(in_data_t), str(out_data_t), str(wt_data_t), str(
            "_" + quantization if quantization != "PACT" else ""), str("_" + folding if folding != "weights" else ""))
        self.filename = self.fn_name + ".c"
        self.quantization = quantization
        self.folding = folding
        self.reordered_no_shift_load_fn = "arm_{0}_to_{1}_reordered".format(str(
            self.in_data_t), self.arithmetic_t)
        self.nn_mat_mul_fn = "arm_nn_mat_mult_kernel_reordered_{0}_{1}_{2}{3}{4}".format(str(wt_data_t),
                                                                                         str(self.arithmetic_t),
                                                                                         str(out_data_t), str(
                "_" + quantization if quantization != "PACT" else ""), str(
                "_" + folding if folding != "weights" else ""))
        self.ch_in_constrain = CMSISConstrains[in_data_t]
        self.ch_out_constrain = CMSISConstrains[out_data_t]
        self.api = self.__class__.__name__

    def generate_code(self):
        return Template(filename="templates/arm_convolve_HWC_x_y_z.c").render(config=self)

    def get_leftover_code(self):
        return ""


class CMSISDepthwise(CMSISFactory):
    def __init__(self, in_data_t, out_data_t, wt_data_t, quantization, folding):
        super().__init__(in_data_t, out_data_t, wt_data_t)
        self.fn_name = "arm_depthwise_separable_conv_HWC_{0}_{1}_{2}{3}{4}".format(str(in_data_t), str(out_data_t), str(wt_data_t), str(
            "_" + quantization if quantization != "PACT" else ""), str("_" + folding if folding != "weights" else ""))
        self.filename = self.fn_name + ".c"
        self.quantization = quantization
        self.folding = folding
        self.api = self.__class__.__name__

    def generate_code(self):
        return Template(filename="templates/arm_depthwise_separable_conv_HWC_x_y_z.c").render(config=self)


class CMSISMatMul(CMSISFactory):
    def __init__(self, out_data_t, wt_data_t, quantization, folding):
        super().__init__("", out_data_t, wt_data_t)
        self.fn_name = "arm_nn_mat_mult_kernel_reordered_{0}_{1}_{2}{3}{4}".format(str(wt_data_t),
                                                                                   str(self.arithmetic_t),
                                                                                   str(out_data_t), str(
                "_" + quantization if quantization != "PACT" else ""), str(
                "_" + folding if folding != "weights" else ""))
        self.filename = self.fn_name + ".c"
        self.quantization = quantization
        self.folding = folding
        self.api = self.__class__.__name__

    def generate_code(self):
        return Template(filename="templates/arm_nn_mat_mult_kernel_reordered_x_y_z.c").render(config=self)


class CMSISConvertReorder(CMSISFactory):
    def __init__(self, in_data_t):
        super().__init__(in_data_t, "", "")
        self.fn_name = "arm_{0}_to_{1}_reordered".format(str(in_data_t), str(self.arithmetic_t))
        self.filename = self.fn_name + ".c"
        self.out_data_t = str(self.arithmetic_t)
        self.api = self.__class__.__name__

    def generate_code(self):
        return Template(filename="templates/arm_x_to_y_reordered.c").render(config=self)


# Generate CMSISConvolve
mkdir_p(CMSISSrcDirs['convolution'])
for i in CMSISDataPrecisions:
    for j in CMSISDataPrecisions:
        for z in CMSISDataPrecisions:
            for q in CMSISQuantizationMethods:
                for f in CMSISFoldingMethods:
                    if (q == "PACT_CH" and f != "weights") or q == "PACT":
                        c = CMSISConvolve(in_data_t=i, out_data_t=j, wt_data_t=z, quantization=q, folding=f)
                        CMSISAPI += c.generate_api() + "\n"
                        new_file = open(CMSISSrcDirs['convolution'] + c.filename, 'w')
                        new_file.write(c.generate_code())
                        new_file.close()

# Generate CMSISDepthwise
mkdir_p(CMSISSrcDirs['convolution'])
for i in CMSISDataPrecisions:
    for j in CMSISDataPrecisions:
        for z in CMSISDataPrecisions:
            for q in CMSISQuantizationMethods:
                for f in CMSISFoldingMethods:
                    if (q == "PACT_CH" and f != "weights") or q == "PACT":
                        c = CMSISDepthwise(in_data_t=i, out_data_t=j, wt_data_t=z, quantization=q, folding=f)
                        CMSISAPI += c.generate_api() + "\n"
                        new_file = open(CMSISSrcDirs['convolution'] + c.filename, 'w')
                        new_file.write(c.generate_code())
                        new_file.close()

# Generate CMSISMatMul
mkdir_p(CMSISSrcDirs['convolution'])
for i in CMSISDataPrecisions:
    for j in CMSISDataPrecisions:
        for q in CMSISQuantizationMethods:
            for f in CMSISFoldingMethods:
                if (q == "PACT_CH" and f != "weights") or q == "PACT":
                    c = CMSISMatMul(out_data_t=i, wt_data_t=j, quantization=q, folding=f)
                    CMSISAPI += c.generate_api() + "\n"
                    new_file = open(CMSISSrcDirs['convolution'] + c.filename, 'w')
                    new_file.write(c.generate_code())
                    new_file.close()

# Generate CMSISConvertReorder
mkdir_p(CMSISSrcDirs['NNSupport'])
for i in CMSISDataPrecisions:
    c = CMSISConvertReorder(in_data_t=i)
    CMSISSupportAPI += c.generate_api() + "\n"
    new_file = open(CMSISSrcDirs['NNSupport'] + c.filename, 'w')
    new_file.write(c.generate_code())
    new_file.close()

# Generate new include files
mkdir_p(CMSISSrcDirs['Include'])
new_file = open(CMSISSrcDirs['Include'] + "/arm_nnfunctions.h", 'w')
new_file.write(Template(filename="templates/arm_nnfunctions.h").render(CMSISAPI=CMSISAPI))
new_file.close()
new_file = open(CMSISSrcDirs['Include'] + "/arm_nnsupportfunctions.h", 'w')
new_file.write(Template(filename="templates/arm_nnsupportfunctions.h").render(CMSISSupportAPI=CMSISSupportAPI))
new_file.close()

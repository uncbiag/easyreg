import SimpleITK as sitk
import os
import numpy as np
import ants
import subprocess
from model_pool.nifty_reg_utils import expand_batch_ch_dim
from model_pool.metrics import get_multi_metric

def __read_and_clean_itk_info(path):
    return sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(path)))

def resize_input_img_and_save_it_as_tmp(img_pth, is_label=False,fname=None,debug_path=None):
        """
        :param img: sitk input, factor is the outputsize/patched_sized
        :return:
        """
        img_org = sitk.ReadImage(img_pth)
        img = __read_and_clean_itk_info(img_pth)
        resampler= sitk.ResampleImageFilter()
        dimension =3
        factor = np.flipud([0.5,0.5,0.5])
        img_sz = img.GetSize()
        affine = sitk.AffineTransform(dimension)
        matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
        after_size = [int(img_sz[i]*factor[i]) for i in range(dimension)]
        after_size = [int(sz) for sz in after_size]
        matrix[0, 0] =1./ factor[0]
        matrix[1, 1] =1./ factor[1]
        matrix[2, 2] =1./ factor[2]
        affine.SetMatrix(matrix.ravel())
        resampler.SetSize(after_size)
        resampler.SetTransform(affine)
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
        img_resampled = resampler.Execute(img)
        fpth = os.path.join(debug_path,fname)
        # img_resampled.SetSpacing(factor_tuple(img_org.GetSpacing(),1./factor))
        # img_resampled.SetOrigin(factor_tuple(img_org.GetOrigin(),factor))
        # img_resampled.SetDirection(img_org.GetDirection())
        sitk.WriteImage(img_resampled, fpth)
        return fpth




# /playpen/zyshen/oai_data/Nifti_rescaled/9357383_20040927_SAG_3D_DESS_LEFT_016610250606_image.nii.gz     /playpen/zyshen/oai_data/Nifti_rescaled/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_image.nii.gz     /playpen/zyshen/oai_data/Nifti_rescaled/9357383_20040927_SAG_3D_DESS_LEFT_016610250606_label_all.nii.gz     /playpen/zyshen/oai_data/Nifti_rescaled/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_label_all.nii.gz
# /playpen/zyshen/oai_data/Nifti_rescaled/9905156_20040816_SAG_3D_DESS_LEFT_016610241205_image.nii.gz     /playpen/zyshen/oai_data/Nifti_rescaled/9369649_20050112_SAG_3D_DESS_LEFT_016610322906_image.nii.gz     /playpen/zyshen/oai_data/Nifti_rescaled/9905156_20040816_SAG_3D_DESS_LEFT_016610241205_label_all.nii.gz     /playpen/zyshen/oai_data/Nifti_rescaled/9369649_20050112_SAG_3D_DESS_LEFT_016610322906_label_all.nii.gz
# /playpen/zyshen/oai_data/Nifti_rescaled/9897397_20041214_SAG_3D_DESS_RIGHT_016610308211_image.nii.gz     /playpen/zyshen/oai_data/Nifti_rescaled/9902757_20051021_SAG_3D_DESS_RIGHT_016610209509_image.nii.gz     /playpen/zyshen/oai_data/Nifti_rescaled/9897397_20041214_SAG_3D_DESS_RIGHT_016610308211_label_all.nii.gz     /playpen/zyshen/oai_data/Nifti_rescaled/9902757_20051021_SAG_3D_DESS_RIGHT_016610209509_label_all.nii.gz
sp_list = ['/playpen/zyshen/oai_data/Nifti_rescaled/9357383_20040927_SAG_3D_DESS_LEFT_016610250606_image.nii.gz',
           '/playpen/zyshen/oai_data/Nifti_rescaled/9905156_20040816_SAG_3D_DESS_LEFT_016610241205_image.nii.gz',
           '/playpen/zyshen/oai_data/Nifti_rescaled/9897397_20041214_SAG_3D_DESS_RIGHT_016610308211_image.nii.gz']

tp_list =['/playpen/zyshen/oai_data/Nifti_rescaled/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_image.nii.gz',
          '/playpen/zyshen/oai_data/Nifti_rescaled/9369649_20050112_SAG_3D_DESS_LEFT_016610322906_image.nii.gz',
          '/playpen/zyshen/oai_data/Nifti_rescaled/9902757_20051021_SAG_3D_DESS_RIGHT_016610209509_image.nii.gz'
          ]

lsp_list= ['/playpen/zyshen/oai_data/Nifti_rescaled/9357383_20040927_SAG_3D_DESS_LEFT_016610250606_label_all.nii.gz',
'/playpen/zyshen/oai_data/Nifti_rescaled/9905156_20040816_SAG_3D_DESS_LEFT_016610241205_label_all.nii.gz',
  '/playpen/zyshen/oai_data/Nifti_rescaled/9897397_20041214_SAG_3D_DESS_RIGHT_016610308211_label_all.nii.gz']


ltp_list = ['/playpen/zyshen/oai_data/Nifti_rescaled/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_label_all.nii.gz',
'/playpen/zyshen/oai_data/Nifti_rescaled/9369649_20050112_SAG_3D_DESS_LEFT_016610322906_label_all.nii.gz',
   '/playpen/zyshen/oai_data/Nifti_rescaled/9902757_20051021_SAG_3D_DESS_RIGHT_016610209509_label_all.nii.gz']
debug_path = '/playpen/zyshen/debugs/demons/'
# for i,fp in enumerate(sp_list):
#     resize_input_img_and_save_it_as_tmp(fp,is_label=False,fname='moving_'+str(i)+'.nii.gz',debug_path=debug_path)
# for i,fp in enumerate(tp_list):
#     resize_input_img_and_save_it_as_tmp(fp,is_label=False,fname='target_'+str(i)+'.nii.gz',debug_path=debug_path)
#
# for i,fp in enumerate(lsp_list):
#     resize_input_img_and_save_it_as_tmp(fp,is_label=True,fname='l_moving'+str(i)+'.nii.gz',debug_path=debug_path)
# for i,fp in enumerate(ltp_list):
#     resize_input_img_and_save_it_as_tmp(fp,is_label=True,fname='l_target'+str(i)+'.nii.gz',debug_path=debug_path)


output_path = '/playpen/zyshen/debugs/zhengyang'
for i,lsp in enumerate(lsp_list):
    ltarget_path = os.path.join(output_path,'l_target'+str(i)+'.nii.gz')
    lmoving_path=os.path.join(output_path,'l_moving'+str(i)+'.nii.gz')
    syn_res = [os.path.join(output_path,'output_'+str(i)+'1Warp.nii.gz'),os.path.join(output_path,'output_'+str(i)+'0GenericAffine.mat')]
    ltarget = ants.image_read(ltarget_path)
    lmoving = ants.image_read(lmoving_path)

    loutput = ants.apply_transforms(fixed=ltarget, moving=lmoving,
                                    transformlist=syn_res,
                                    interpolator='nearestNeighbor')
    func ='/playpen/zyshen/ITKTransformTools/install/bin'
    st_pth1 = '9357383_20040927_SAG_3D_DESS_LEFT_016610250606_image_9003406_20060322_SAG_3D_DESS_LEFT_016610899303_image_affine.mat'
    st_pth2 = '9357383_20040927_SAG_3D_DESS_LEFT_016610250606_image_9003406_20060322_SAG_3D_DESS_LEFT_016610899303_image_affine.mat'
    cmd =func +'  concatenate test_output.nii -r target_0.nii.gz '+ st_pth1 +' ' +st_pth2+ ' displacement'
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    # loutput = ants.apply_transforms(fixed=ltarget, moving=lmoving,
    #                                 transformlist=['/playpen/zyshen/debugs/zhengyang/test_output.nii.gz'],
    #                                 interpolator='nearestNeighbor')
    #res = ants.create_jacobian_determinant_image(ltarget, syn_res[0], do_log=False, geom=False)

    loutput = loutput.numpy()
    loutput = np.transpose(loutput, (2, 1, 0))
    loutput_np=expand_batch_ch_dim(loutput)
    ltarget = ltarget.numpy()
    ltarget = np.transpose(ltarget, (2, 1, 0))
    ltarget_np = expand_batch_ch_dim(ltarget)

    val_res_dic = get_multi_metric(loutput_np, ltarget_np, rm_bg=False)
    res =np.mean(val_res_dic['batch_avg_res']['dice'][0,1:])
    print("the dice score of the pair {} is {}".format(i, res))



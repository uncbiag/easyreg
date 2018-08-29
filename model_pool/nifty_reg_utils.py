import os
import json
import sys
import subprocess
import nibabel as nib
import numpy as np
# from model_pool.task_pool_reg import is_llm
#
# if not is_llm:
#     nifty_bin = '/playpen/zyshen/package/niftyreg-git/niftyreg_install/bin'
# else:
#     nifty_bin = '/playpen/raid/zyshen/package/niftyreg-git/niftyreg_install/bin'
nifty_bin = '/playpen/raid/zyshen/package/niftyreg-git/niftyreg_install/bin'


def nifty_reg_bspline(ref, flo, res=None, cpp=None, rmask=None, fmask=None, levels=None,aff= None):
    executable = nifty_bin + '/reg_f3d'

    cmd = executable + ' -ref ' + ref + ' -flo ' + flo
    if cpp != None:
        cmd += ' -cpp ' + cpp
    if res != None:
        cmd += ' -res ' + res

    if aff !=None:
        cmd += ' -aff ' + aff
    if rmask != None:
        cmd += ' -rmask ' + rmask
    if fmask != None:
        cmd += ' -fmask ' + fmask
    if levels != None:
        cmd += ' -lp ' + str(levels)
    cmd = cmd + ' -sx -10 --lncc 40 -pad 0 '
    #    cmd = cmd + ' -sx 10 --nmi --rbn 100 --fbn 100 -gpu -pad 0 -pert 1'

    return cmd


def nifty_reg_affine(ref, flo, res=None, aff=None, rmask=None, fmask=None, symmetric=True, init='center'):
    executable = nifty_bin + '/reg_aladin'
    cmd = executable + ' -ref ' + ref + ' -flo ' + flo
    if res != None:
        cmd += ' -res ' + res
    if aff != None:
        cmd += ' -aff ' + aff
    if rmask != None:
        cmd += ' -rmask ' + rmask
    if fmask != None:
        cmd += ' -fmask ' + fmask
    if symmetric == None:
        cmd += ' -noSym'
    #    if init != 'center':
    #        cmd += ' -' + init
    return cmd


def nifty_reg_transform(ref=None, ref2=None, invAff1=None, invAff2=None, invNrr1=None, invNrr2=None,
                        invNrr3=None, disp1=None, disp2=None, def1=None, def2=None, comp1=None, comp2=None,
                        comp3=None):
    executable = nifty_bin + '/reg_transform'
    cmd = executable
    if ref != None:
        cmd += ' -ref ' + ref
    if ref2 != None:
        cmd += ' -ref2 ' + ref2
    if invAff1 != None and invAff2 != None:
        cmd += ' -invAff ' + invAff1 + ' ' + invAff2
    elif disp1 != None and disp2 != None:
        cmd += ' -disp ' + disp1 + ' ' + disp2
    elif def1 != None and def2 != None:
        cmd += ' -def ' + def1 + ' ' + def2
    elif comp1 != None and comp2 != None and comp3 != None:
        cmd += ' -comp ' + comp1 + ' ' + comp2 + ' ' + comp3
    elif invNrr1 != None and invNrr2 != None and invNrr3 != None:
        cmd += ' -invNrr ' + invNrr1 + ' ' + invNrr2 + ' ' + invNrr3

    return cmd


def nifty_reg_resample(ref, flo, trans=None, res=None, inter=None):
    executable = nifty_bin + '/reg_resample'
    cmd = executable + ' -ref ' + ref + ' -flo ' + flo
    if trans != None:
        cmd += ' -trans ' + trans
    if res != None:
        cmd += ' -res ' + res
    if inter != None:
        cmd +=' -inter ' + str(inter)

    return cmd



def nifty_read(path):
    img = nib.load(path)
    img_data =  img.get_fdata()
    return np.transpose(img_data)
def expand_batch_ch_dim(input):
    return np.expand_dims(np.expand_dims(input,0),0)


def performRegistration(mv_path, target_path, registration_type='bspline', record_path = None, ml_path=None):
    if record_path is None:
        record_path = './'
    deformation_path = os.path.join(record_path, 'deformation.nii')
    displacement_path = os.path.join(record_path, 'displacement.nii')
    affine_path = os.path.join(record_path, 'affine_image.nii.gz')
    bspline_path = os.path.join(record_path, 'bspline_image.nii.gz')
    output_path = None
    loutput =None
    cmd = ""
    if True:
        affine_txt = os.path.join(record_path, 'affine_transform.txt')
        cmd += '\n' + nifty_reg_affine(ref=target_path, flo=mv_path, aff=affine_txt, res=affine_path)
        output_path = affine_path
        output_txt = affine_txt
    if registration_type =='bspline':
        bspline_txt = os.path.join(record_path,'bspline_transform.nii')
        cmd += '\n' + nifty_reg_bspline(ref=target_path, flo=mv_path, cpp=bspline_txt, res=bspline_path, aff= affine_txt )
        output_path = bspline_path
        output_txt = bspline_txt

    # cmd += '\n' + nifty_reg_transform(ref=target_path, def1=output_txt, def2=deformation_path)
    #cmd += '\n' + nifty_reg_transform(ref=target_path, disp1=output_txt, disp2=displacement_path)
    if ml_path is not None:
        loutput_path = os.path.join(record_path, 'warped_label.nii.gz')
        cmd += '\n' + nifty_reg_resample(ref=target_path,flo=ml_path,trans=output_txt, res=loutput_path, inter= 0)

    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    output = nifty_read(output_path)
    #phi = nifty_read(displacement_path)
    if ml_path:
        loutput = nifty_read(loutput_path)

    return expand_batch_ch_dim(output), expand_batch_ch_dim(loutput), None
import os
import subprocess
import numpy as np
import nibabel as nib


def nifty_reg_bspline(nifty_bin, nifty_reg_cmd, ref, flo, res=None, cpp=None, rmask=None, fmask=None, levels=None,aff= None):
    """
    call bspline registration in niftyreg

    :param nifty_bin: the path of niftyreg bin
    :param nifty_reg_cmd: nifty reg bspline command
    :param ref:
    :param flo:
    :return:
    """
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
    cmd = cmd +  nifty_reg_cmd #' -pad 0 '  #' -sx -10 --lncc 40 -pad 0 '
    return cmd


def nifty_reg_affine(nifty_bin, ref, flo, res=None, aff=None, rmask=None, fmask=None, symmetric=True, init='center'):
    """
    call affine registration in niftyreg

    :param nifty_bin: the path of niftyreg bin
    :param ref:
    :param flo:
    :return:
    """
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


def nifty_reg_transform(nifty_bin, ref=None, ref2=None, invAff1=None, invAff2=None, invNrr1=None, invNrr2=None,
                        invNrr3=None, disp1=None, disp2=None, def1=None, def2=None, comp1=None, comp2=None,
                        comp3=None):
    """
    call niftyreg transform, see niftyreg doc, http://cmictig.cs.ucl.ac.uk/wiki/index.php/Reg_transform

    :param nifty_bin: the path of nifityreg bin
    :return:
    """
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


def nifty_reg_resample(nifty_bin, ref, flo, trans=None, res=None, inter=None, pad=0):
    """
    call niftyreg resample http://cmictig.cs.ucl.ac.uk/wiki/index.php/Reg_resample

    :param nifty_bin:  the path of nifityreg bin
    :param ref:
    :param flo:
    :return:
    """
    executable = nifty_bin + '/reg_resample'
    cmd = executable + ' -ref ' + ref + ' -flo ' + flo
    if trans != None:
        cmd += ' -trans ' + trans
    if res != None:
        cmd += ' -res ' + res
    if inter != None:
        cmd +=' -inter ' + str(inter)
    if pad !=0:
        cmd +=' -pad ' + str(pad)

    return cmd


def nifty_reg_jacobian(nifty_bin, ref, trans=None, res=None):
    """
    call nifty reg compute jacobian

    :param nifty_bin: the path of niftyreg bin
    :param ref: path of reference image
    :param trans: path of transformation
    :param res: path of output
    :return:
    """
    executable = nifty_bin + '/reg_jacobian'
    cmd = executable + ' -trans ' + trans + ' -ref ' + ref + ' -jac ' + res
    return cmd



def nifty_read(path):
    """
    read image

    :param path:
    :return:
    """
    img = nib.load(path)
    img_data =  img.get_fdata()
    return np.transpose(img_data)
def expand_batch_ch_dim(input):
    """
    expand dimension [1,1] +[x,y,z]

    :param input: numpy array
    :return:
    """
    if input is not None:
        return np.expand_dims(np.expand_dims(input,0),0)
    else:
        return None





def performRegistration(param, mv_path, target_path, registration_type='bspline', record_path = None, ml_path=None,fname = ''):
    """
    call niftyreg image registration

    :param param: ParameterDict, settings for niftyreg
    :param mv_path: path of moving image
    :param target_path: path of target image
    :param registration_type: 'affine' or 'bspline'
    :param record_path: path of saving path
    :param ml_path: path of moving label
    :param fname: pair name
    :return:
    """
    if record_path is None:
        record_path = './'
    nifty_bin = param[('nifty_bin','','path of the niftyreg bin file')]
    nifty_reg_cmd = param[('nifty_reg_cmd','','command line for nifityreg registration')]
    assert os.path.exists(nifty_bin), "The niftyreg is not detected, please set the niftyreg binary path"
    deformation_path = os.path.join(record_path,fname+ '_deformation.nii.gz')
    affine_path = os.path.join(record_path, fname+ '_affine_image.nii.gz')
    bspline_path = os.path.join(record_path, fname+ '_bspline_image.nii.gz')
    jacobi_path = os.path.join(record_path, fname+ '_jacobi_image.nii.gz')
    output_path = None
    loutput =None
    cmd = ""
    output_txt = None
    if True :
        affine_txt = os.path.join(record_path, fname+'_affine_transform.txt')
        cmd += '\n' + nifty_reg_affine(nifty_bin=nifty_bin, ref=target_path, flo=mv_path, aff=affine_txt, res=affine_path)
        output_path = affine_path
        output_txt = affine_txt
    if registration_type =='bspline':
        bspline_txt = os.path.join(record_path,fname+'_bspline_transform.nii')
        cmd += '\n' + nifty_reg_bspline(nifty_bin=nifty_bin, nifty_reg_cmd=nifty_reg_cmd, ref=target_path, flo=mv_path, cpp=bspline_txt, res=bspline_path, aff= output_txt )
        output_path = bspline_path
        output_txt = bspline_txt
        cmd += '\n' + nifty_reg_jacobian(nifty_bin=nifty_bin, ref=target_path, trans=output_txt, res=jacobi_path)

    cmd += '\n' + nifty_reg_transform(nifty_bin=nifty_bin,ref=target_path, def1=output_txt, def2=deformation_path)
    if ml_path is not None:
        loutput_path = os.path.join(record_path, fname+'_warped_label.nii.gz')
        cmd += '\n' + nifty_reg_resample(nifty_bin=nifty_bin, ref=target_path,flo=ml_path,trans=output_txt, res=loutput_path, inter= 0)

    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    output = nifty_read(output_path)
    phi = None # nifty_read_phi(deformation_path)
    jacobi = None
    if registration_type == 'bspline':
        jacobi =nifty_read(jacobi_path)

    if ml_path:
        loutput = nifty_read(loutput_path)

    return expand_batch_ch_dim(output), expand_batch_ch_dim(loutput), phi,jacobi
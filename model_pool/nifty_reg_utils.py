import os
import subprocess
import numpy as np
import SimpleITK as sitk
import nibabel as nib


def nifty_reg_bspline(nifty_bin, nifty_reg_cmd, ref, flo, res=None, cpp=None, rmask=None, fmask=None, levels=None,aff= None):
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
    #    cmd = cmd + ' -sx 10 --nmi --rbn 100 --fbn 100 -gpu -pad 0 -pert 1'

    return cmd


def nifty_reg_affine(nifty_bin, ref, flo, res=None, aff=None, rmask=None, fmask=None, symmetric=True, init='center'):
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
    executable = nifty_bin + '/reg_jacobian'
    cmd = executable + ' -trans ' + trans + ' -ref ' + ref + ' -jac ' + res
    return cmd



def nifty_read(path):
    img = nib.load(path)
    img_data =  img.get_fdata()
    return np.transpose(img_data)
def expand_batch_ch_dim(input):
    return np.expand_dims(np.expand_dims(input,0),0)

def nifty_read_phi(path):
    phi_nib = nib.load(path)
    phi = phi_nib.get_fdata()
    phi_tmp = np.zeros([1,3]+list(phi.shape[:3]))
    phi_tmp[0,0] = -phi[...,0,0]
    phi_tmp[0,1] = -phi[...,0,1]
    phi_tmp[0,2] =  phi[...,0,2]
    return phi_tmp



def init_phi(mv_path,phi_path='.',record_path=None):
     mv = sitk.ReadImage(mv_path)
     x = sitk.ReadImage(os.path.join(phi_path,'identity_x.nii.gz'))
     y = sitk.ReadImage(os.path.join(phi_path,'identity_y.nii.gz'))
     z = sitk.ReadImage(os.path.join(phi_path,'identity_z.nii.gz'))
     assert x.GetSize() == mv.GetSize()
     assert y.GetSize() == mv.GetSize()
     assert z.GetSize() == mv.GetSize()
     x.SetSpacing(mv.GetSpacing())
     x.SetOrigin(mv.GetOrigin())
     x.SetDirection(mv.GetDirection())
     y.SetSpacing(mv.GetSpacing())
     y.SetOrigin(mv.GetOrigin())
     y.SetDirection(mv.GetDirection())
     z.SetSpacing(mv.GetSpacing())
     z.SetOrigin(mv.GetOrigin())
     z.SetDirection(mv.GetDirection())
     sitk.WriteImage(x,os.path.join(record_path,'identity_x.nii.gz'))
     sitk.WriteImage(y,os.path.join(record_path,'identity_y.nii.gz'))
     sitk.WriteImage(z,os.path.join(record_path,'identity_z.nii.gz'))

def _get_deformation(nifty_bin, cmd, target_path, record_path,output_txt):
    cmd += '\n' + nifty_reg_resample(nifty_bin=nifty_bin, ref=target_path, flo=os.path.join(record_path,'identity_x.nii.gz'), trans=output_txt, res=os.path.join(record_path,'warped_x.nii.gz'), inter=1)
    cmd += '\n' + nifty_reg_resample(nifty_bin=nifty_bin, ref=target_path, flo=os.path.join(record_path,'identity_y.nii.gz'), trans=output_txt, res=os.path.join(record_path,'warped_y.nii.gz'), inter=1)
    cmd += '\n' + nifty_reg_resample(nifty_bin=nifty_bin, ref=target_path, flo=os.path.join(record_path,'identity_z.nii.gz'), trans=output_txt, res=os.path.join(record_path,'warped_z.nii.gz'), inter=1)
    return cmd

def combine_deformation(record_path):
    x = sitk.ReadImage(os.path.join(record_path,'warped_x.nii.gz'))
    y = sitk.ReadImage(os.path.join(record_path,'warped_y.nii.gz'))
    z = sitk.ReadImage(os.path.join(record_path,'warped_z.nii.gz'))
    x_np = sitk.GetArrayFromImage(x)
    y_np = sitk.GetArrayFromImage(y)
    z_np = sitk.GetArrayFromImage(z)
    xyz = np.zeros([1,3]+list(x_np.shape))
    xyz[0, 0] = x_np
    xyz[0, 1] = y_np
    xyz[0, 2] = z_np
    phi = nib.Nifti1Image(xyz[0],np.eye(4))
    nib.save(phi,os.path.join(record_path,'phi.nii.gz'))
    return xyz





def performRegistration(param, mv_path, target_path, registration_type='bspline', record_path = None, ml_path=None, affine_on=True,fname = ''):
    if record_path is None:
        record_path = './'
    nifty_bin = param['nifty_bin']
    nifty_reg_cmd = param['nifty_reg_cmd']
    deformation_path = os.path.join(record_path,fname+ '_deformation.nii.gz')
    displacement_path = os.path.join(record_path, fname+ '_displacement.nii.gz')
    affine_path = os.path.join(record_path, fname+ '_affine_image.nii.gz')
    bspline_path = os.path.join(record_path, fname+ '_bspline_image.nii.gz')
    jacobi_path = os.path.join(record_path, fname+ '_jacobi_image.nii.gz')
    output_path = None
    loutput =None
    #init_phi(mv_path,phi_path='.',record_path=record_path)
    cmd = ""
    output_txt = None
    if True and affine_on :
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
    #cmd = _get_deformation(cmd, target_path,record_path,output_txt)
    #cmd += '\n' + nifty_reg_transform(ref=target_path, disp1=output_txt, disp2=displacement_path)

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
    #print("starting merge phi")
    #phi = combine_deformation(record_path)

    if ml_path:
        loutput = nifty_read(loutput_path)

    return expand_batch_ch_dim(output), expand_batch_ch_dim(loutput), phi,jacobi
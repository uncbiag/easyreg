import os
import numpy as np
import unittest
import numpy.testing as npt
import subprocess

class Test_Networks(unittest.TestCase):


    def setUp(self):
        self.image_path = None
        self.python_executable = 'python'
        cur_dir = os.getcwd()
        par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
        self.demo_dir = os.path.join(par_dir,'demo')
        self.gpu_id  = 0
        self.input_pair_txt_path = '/playpen/zyshen/debugs/get_val_and_debug_res/example_for_demo.txt'
        self.output_folder = os.path.join(cur_dir,'test_networks_ouput')
        os.makedirs(self.output_folder,exist_ok=True)

    def get_experiment_data_from_record_detail(self, path):
        data_detail = np.load(path)
        data = np.mean(data_detail[:, 1:], 1)
        return data

    def tearDown(self):
        pass

    def test_vsvf_network(self):
        output_path = os.path.join(self.output_folder,'test_vsvf_net')
        cmd = '{} demo_for_easyreg_eval.py --run_demo --demo_name=network_vsvf -txt={} -o={} -g={}'.\
            format(self.python_executable, self.input_pair_txt_path, output_path, self.gpu_id)
        process = subprocess.Popen(cmd, cwd=self.demo_dir, shell=True)
        process.wait()
        res_np_path = os.path.join(output_path, 'reg/res/records/records_detail.npy')
        dice_score_np = self.get_experiment_data_from_record_detail(res_np_path)
        dice_score_avg = np.mean(dice_score_np)
        npt.assert_almost_equal(dice_score_avg, 0.6637740976354091, decimal=4)

    def test_rdmm_network(self):
        output_path = os.path.join(self.output_folder,'test_rdmm_net')
        cmd = '{} demo_for_easyreg_eval.py --run_demo --demo_name=network_rdmm -txt={} -o={} -g={}'. \
            format(self.python_executable, self.input_pair_txt_path, output_path, self.gpu_id)
        process = subprocess.Popen(cmd, cwd=self.demo_dir, shell=True)
        process.wait()
        res_np_path = os.path.join(output_path, 'reg/res/records/records_detail.npy')
        dice_score_np = self.get_experiment_data_from_record_detail(res_np_path)
        dice_score_avg = np.mean(dice_score_np)
        npt.assert_almost_equal(dice_score_avg, 0.6695723796958148, decimal=4)




def run_by_name(test_name):
    suite = unittest.TestSuite()
    suite.addTest(Test_Networks(test_name))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
       run_by_name('test_vsvf_network')
       run_by_name('test_rdmm_network')


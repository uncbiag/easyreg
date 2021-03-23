import os
import subprocess
import unittest

import numpy as np
import numpy.testing as npt


class Test_Networks(unittest.TestCase):


    def setUp(self):
        self.image_path = None
        self.python_executable = 'python'
        self.cur_dir = os.getcwd()
        par_dir = os.path.abspath(os.path.join(self.cur_dir, os.pardir))
        self.demo_dir = os.path.join(par_dir,'demo')
        self.gpu_id  = 0
        self.input_pair_txt_path = os.path.join(self.demo_dir,'oai_examples.txt')
        assert os.path.isfile(self.input_pair_txt_path), "to run this test, please download the demo material first!"
        self.output_folder = os.path.join(self.cur_dir,'test_networks_ouput')
        os.makedirs(self.output_folder,exist_ok=True)

    def get_experiment_data_from_record_detail(self, path):
        data_detail = np.load(path)
        data = np.mean(data_detail[:, 1:], 1)
        return data

    def tearDown(self):
        pass

    def test_vsvf_network(self):
        output_path = os.path.join(self.output_folder,'test_vsvf_net')
        task_setting_folder = os.path.join(self.cur_dir,'settings',"eval_network_vsvf")
        cmd = '{} demo_for_easyreg_eval.py -ts={} -txt={} -o={} -g={}'.\
            format(self.python_executable, task_setting_folder, self.input_pair_txt_path, output_path, self.gpu_id)
        process = subprocess.Popen(cmd, cwd=self.demo_dir, shell=True)
        process.wait()
        res_np_path = os.path.join(output_path, 'reg/res/records/records_detail.npy')
        dice_score_np = self.get_experiment_data_from_record_detail(res_np_path)
        dice_score_avg = np.mean(dice_score_np)
        npt.assert_almost_equal(dice_score_avg, 0.6631099637937586, decimal=4)

    def test_rdmm_network(self):
        output_path = os.path.join(self.output_folder,'test_rdmm_net')
        task_setting_folder = os.path.join(self.cur_dir,'settings',"eval_network_rdmm")
        cmd = '{} demo_for_easyreg_eval.py -ts={} -txt={} -o={} -g={}'. \
            format(self.python_executable, task_setting_folder, self.input_pair_txt_path, output_path, self.gpu_id)
        process = subprocess.Popen(cmd, cwd=self.demo_dir, shell=True)
        process.wait()
        res_np_path = os.path.join(output_path, 'reg/res/records/records_detail.npy')
        dice_score_np = self.get_experiment_data_from_record_detail(res_np_path)
        dice_score_avg = np.mean(dice_score_np)
        npt.assert_almost_equal(dice_score_avg, 0.6695689292780935, decimal=4)




def run_by_name(test_name):
    suite = unittest.TestSuite()
    suite.addTest(Test_Networks(test_name))
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
       run_by_name('test_vsvf_network')
       run_by_name('test_rdmm_network')


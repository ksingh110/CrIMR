import numpy as np
mutated_data = np.load("/Users/krishaysingh/Documents/CrIMR/E:\datasets\processeddata\MUTATION_DATA_TRAINING_100.npz", allow_pickle=True)
mutated_test1 = mutated_data['arr_0'][0]
output_test = "TEST_1_M"
np.savez_compressed(output_test, arr_0=np.array(mutated_test1))
mutated_test2 = mutated_data['arr_0'][1]
output_test2 = "TEST_2_M"
np.savez_compressed(output_test2, arr_0=np.array(mutated_test2))
mutated_test3 = mutated_data['arr_0'][2]
output_test3 = "TEST_3_M"
np.savez_compressed(output_test3, arr_0=np.array(mutated_test3))

mutated_test4 = mutated_data['arr_0'][3]
output_test4 = "TEST_4_M"
np.savez_compressed(output_test4, arr_0=np.array(mutated_test4))
mutated_test5 = mutated_data['arr_0'][4]
output_test5 = "TEST_5_M"
np.savez_compressed(output_test5, arr_0=np.array(mutated_test5))
mutated_test6 = mutated_data['arr_0'][5]
output_test6 = "TEST_6_M"
np.savez_compressed(output_test6, arr_0=np.array(mutated_test6))

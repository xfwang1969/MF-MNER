'''
@Author: HAOZE DU, JIAHAO XV
@Date: 2023-09-27
@LastEditTime: 2023-10-6
@Description: This file is for building model. 
@All Right Reserve


import json

FILE1 = './CCKS_2019_Task1/subtask1_training_part1.txt'
FILE2 = './CCKS_2019_Task1/subtask1_training_part2.txt'
FILE3 = './CCKS_2019_Task1/subtask1_test_set_with_answer.json'

PATH1 = './CCKS_2019_Task1/data/data1-'
PATH2 = './CCKS_2019_Task1/data/data2-'
PATH3 = './CCKS_2019_Task1/data_test/data-test-'

def Process_File(FILENAME, PATH, enc):
  with open(FILENAME, 'r', encoding=enc) as f:
      i = 0
      while True:
        txt = f.readline()
        if not txt: break         # end loop
        i+= 1
        j = json.loads(txt)
        orig = j['originalText'] 
        entities = j['entities']  
        pathO = PATH + str(i) + '-original.txt'
        pathE = PATH + str(i) + '.txt'

        with open(pathO, 'w', encoding='utf-8') as o1: 
            o1.write(orig)
            o1.flush

        with open(pathE, 'w', encoding='utf-8') as o2: 
            for e in entities:
              start = e['start_pos']   
              end = e['end_pos']       
              name = orig[start:end]   
              ty = e['label_type']     
              label = '{0}\t{1}\t{2}\t{3}\n'.format(name, start, end, ty)
              o2.write(label)
              o2.flush

#%%
#
Process_File(FILE1, PATH1, 'utf-8-sig')#
Process_File(FILE2, PATH2, 'utf-8-sig')
Process_File(FILE3, PATH3, 'utf-8')

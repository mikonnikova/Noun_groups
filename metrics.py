import pickle
import sys


# compare group found for the certain noun with a correct answer
# return True if similar, False otherwise

def compare_group(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


# compare given dictionary of groups to a correct one
# return true_positive count, retrieved instances count, relevant instances count for IR defined metrics

def compare(groups, answers):
    correct = 0
    
    for k,v in answers.items():
        if k in groups:
            if compare_group(groups[k], v):
                correct += 1
                
    return correct, len(groups), len(answers)  

	
# compare given list of groups to a correct one
# return precision, recall and f1 metrics (IR defined) for one sentence

def local_metrics(true_positive, found, positive):	

    if positive == 0:
        if found == 0:
            precision, recall, f1 = 1, 1, 1
        else:
            precision, recall, f1 = 0, 1, 0
    else:
        if true_positive == 0:
            precision, recall, f1 = 1, 0, 0
        else:
            precision = true_positive / found
            recall = true_positive / positive
            f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
	

# measure quality of given answers compared to correct ones
# return precision, recall and f1 metrics (IR defined), micro and macro measured

def metrics(given_answers_file, answers_file):
    all_true_positive, all_found, all_positive = 0, 0, 0
    total_precision, total_recall, total_f1 = 0, 0, 0
    count = 0

    with open(given_answers_file, 'rb') as f, open(answers_file, 'rb') as af:
        while True:  # read given_answers_file and answers_file line by line
            try:
                line = pickle.load(f)
            except EOFError:    # end of given_answers_file
                try:
                    _ = pickle.load(af)
                except EOFError:
                    break
                print('Answers file too long; ' + str(count) + ' lines read\n')
                break
            try:
                answer = pickle.load(af)
            except EOFError:    # end of answers_file
                print('Answers file too short; ' + str(count) + ' lines read\n')
                break

            true_positive, found, positive = compare(line, answer)
            local_precision, local_recall, local_f1 = local_metrics(true_positive, found, positive)
			
            all_true_positive += true_positive
            all_found += found
            all_positive += positive
			
            total_precision += local_precision
            total_recall += local_recall
            total_f1 += local_f1
			
            count += 1

    if count == 0:
        print('No lines in file!\n')
        return 0, 0, 0

    macro_precision = all_true_positive / all_found if all_found > 0 else \
        0 if all_true_positive - all_found > 0 else 1
    macro_recall = all_true_positive / all_positive if all_positive > 0 else \
        0 if all_true_positive - all_positive > 0 else 1
    macro_f1 = 2*macro_precision*macro_recall / (macro_precision+macro_recall) \
        if macro_precision+macro_recall > 0 else 0
	
    micro_precision = total_precision / count
    micro_recall = total_recall / count
    micro_f1 = total_f1 / count
	
    print('Macro precision: ' + str(macro_precision))
    print('Macro recall: ' + str(macro_recall))
    print('Macro f1: ' + str(macro_f1))
    print('Micro precision: ' + str(micro_precision))
    print('Micro recall: ' + str(micro_recall))
    print('Micro f1: ' + str(micro_f1))
	
    return micro_f1, macro_f1


if __name__ == '__main__':
    answers_file = sys.argv[1]
    given_answers_file = sys.argv[2]
    metrics(given_answers_file, answers_file)

import pickle


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

# measure quality of given answers compared to correct ones
# return precision, recall and f1 metrics (IR defined)

def metrics(given_answers_file, answers_file):
    all_true_positive, all_found, all_positive = 0, 0, 0
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
            all_true_positive += true_positive
            all_found += found
            all_positive += positive
            count += 1

    if count == 0:
        print('No lines in file!\n')
        return 0, 0, 0

    precision = all_true_positive / all_positive if all_positive > 0 else \
        0 if all_true_positive - all_positive > 0 else 1
    recall = all_true_positive / all_found if all_found > 0 else \
        0 if all_true_positive - all_found > 0 else 1
    f1 = 2*precision*recall / (precision+recall) if precision+recall > 0 else 0
    return precision, recall, f1


# usage example
# answers_file = './Answers/test.txt'
# given_answers_file = './Answers/given.txt'
# print(metrics(given_answers_file, answers_file))

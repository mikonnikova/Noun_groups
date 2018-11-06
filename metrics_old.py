import pickle


# find a group similar to a given one in a correct groups list
# return True if found, False otherwise

def find_similar(group, answers):
    for answer in answers:
        if answer[0] == group[0] and len(answer) == len(group):
            res = True
            for i in range(len(answer)):
                if answer[i] != group[i]:
                    res = False
                    break
            if res:
                return True

    return False


# compare given list of groups to a correct one
# return precision, recall and f1 metrics (IR defined)

def count_metrics(groups, answers):
    groups_count = len(groups)
    answers_count = len(answers)
    similar_count = 0

    for group in groups:
        if find_similar(group, answers):
            similar_count += 1

    precision = similar_count / groups_count if groups_count > 0 else 0
    recall = similar_count / answers_count if answers_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


# measure quality of given answers compared to correct ones
# return overall precision, recall and f1 metrics (IR defined)

def metrics(given_answers_file, answers_file):
    overall_precision = 0
    overall_recall = 0
    overall_f1 = 0
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

            precision, recall, f1 = count_metrics(line, answer)
            overall_precision += precision
            overall_recall += recall
            overall_f1 += f1
            count += 1

    if count == 0:
        print('No lines in file!\n')
        return 0, 0, 0

    precision = overall_precision / count
    recall = overall_recall / count
    f1 = overall_f1 / count
    return precision, recall, f1


# usage example
# answers_file = './Answers/test.txt'
# given_answers_file = './Answers/given.txt'
# print(metrics(given_answers_file, answers_file))
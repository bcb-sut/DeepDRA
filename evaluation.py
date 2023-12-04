from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, precision_score, \
    recall_score, accuracy_score

from utils import *


class Evaluation:
    @staticmethod
    def plot_train_val_accuracy(train_accuracies, val_accuracies, num_epochs):
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('h')
        plt.plot(range(1, num_epochs + 1), train_accuracies)
        plt.plot(range(1, num_epochs + 1), val_accuracies)
        plt.show()

    @staticmethod
    def plot_train_val_loss(train_loss, val_loss, num_epochs):
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('h')
        plt.plot(range(1, num_epochs + 1), train_loss)
        plt.plot(range(1, num_epochs + 1), val_loss)
        plt.show()

    @staticmethod
    def evaluate(all_targets, mlp_output, show_plot=True):
        predicted_labels = np.where(mlp_output > 0.5, 1, 0)
        # Collect predictions and targets for later evaluation
        predicted_labels = predicted_labels.reshape(-1)
        # Convert predictions and targets to numpy arrays
        all_predictions = predicted_labels

        # Calculate and print AUC
        fpr, tpr, thresholds = metrics.roc_curve(all_targets, mlp_output)
        auc = np.round(metrics.auc(fpr, tpr), 2)

        # Calculate and print AUPRC
        print(all_targets)
        precision, recall, thresholds = metrics.precision_recall_curve(all_targets, mlp_output)
        auprc = np.round(metrics.auc(recall, precision), 2)
        # auprc = average_precision_score(all_targets, mlp_output)

        print('Accuracy: {:.2f}'.format(np.round(accuracy_score(all_targets, all_predictions), 2)))
        print('AUC: {:.2f}'.format(auc))
        print('AUPRC: {:.2f}'.format(auprc))

        # Calculate and print confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        accuracy = cm.trace() / np.sum(cm)
        precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
        f1_score = 2 * precision * recall / (precision + recall)
        print('Confusion matrix:\n', cm, sep='')
        print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 score: {f1_score:.3f}')

        if show_plot:
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: AUC={auc}')
            plt.plot(fpr, tpr)
            plt.show()
            # print(f'AUC: {auc}')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR Curve: AUPRC={auprc}')
            plt.plot(recall, precision)
            plt.show()

            prediction_targets = pd.DataFrame({}, columns=['Prediction', 'Target'])

            res = pd.concat(
                [pd.DataFrame(mlp_output.numpy(), ), pd.DataFrame(all_targets.numpy())], axis=1,
                ignore_index=True)

            res.columns = prediction_targets.columns
            prediction_targets = pd.concat([prediction_targets, res])

            class_one = prediction_targets.loc[prediction_targets['Target'] == 0, 'Prediction'].astype(
                np.float32).tolist()
            class_minus_one = prediction_targets.loc[prediction_targets['Target'] == 1, 'Prediction'].astype(
                np.float32).tolist()

            fig, ax = plt.subplots()
            ax.set_ylabel("DeepDRA score")
            xticklabels = ['Responder', 'Non Responder']
            ax.set_xticks([1, 2])
            ax.set_xticklabels(xticklabels)
            data_to_plot = [class_minus_one, class_one]
            plt.ylim(0, 1)
            p_value = np.format_float_scientific(ttest_ind(class_one, class_minus_one)[1])
            cancer = 'all'
            plt.title(
                f'Responder/Non responder scores for {cancer} cancer with \np-value ~= {p_value[0]}e{p_value[-3:]} ')
            bp = ax.violinplot(data_to_plot, showextrema=True, showmeans=True, showmedians=True)
            bp['cmeans'].set_color('r')
            bp['cmedians'].set_color('g')
            plt.show()

        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 score': f1_score, 'AUC': auc,
                'AUPRC': auprc}

    @staticmethod
    def add_results(result_list, current_result):
        result_list['AUC'].append(current_result['AUC'])
        result_list['AUPRC'].append(current_result['AUPRC'])
        result_list['Accuracy'].append(current_result['Accuracy'])
        result_list['Precision'].append(current_result['Precision'])
        result_list['Recall'].append(current_result['Recall'])
        result_list['F1 score'].append(current_result['F1 score'])
        return result_list

    @staticmethod
    def show_final_results(result_list):
        print("Final Results:")
        for i in range(len(result_list["AUC"])):
            accuracy = result_list['Accuracy'][i]
            precision = result_list['Precision'][i]
            recall = result_list['Recall'][i]
            f1_score = result_list['F1 score'][i]
            auc = result_list['AUC'][i]
            auprc = result_list['AUPRC'][i]

            print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 score: {f1_score:.3f}, AUC: {auc:.3f}, ,AUPRC: {auprc:.3f}')

        avg_auc = np.mean(result_list['AUC'])
        avg_auprc = np.mean(result_list['AUPRC'])
        std_auprc = np.std(result_list['AUPRC'])
        print(" Average AUC: {:.3f} \t Average AUPRC: {:.3f} \t Std AUPRC: {:.3f}".format(avg_auc, avg_auprc, std_auprc))

import matplotlib.pyplot as plt

class Visualization:
    @staticmethod
    def results(Y_test, Y_pred, model_type, mse):
        plt.figure(figsize=(7,6))
        
        if model_type == "1":
            color = 'blue'
            tittle = f'Árvore de Decisão (MSE: {mse:.2f})'
        else:
            color = 'red'
            tittle = f'Naive Bayes (MSE: {mse:.2f})'
            
        plt.scatter(Y_test, Y_pred, alpha=0.7, color=color)
        plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=3)
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Previstos')
        plt.title(tittle)
        
        plt.tight_layout()
        plt.show()
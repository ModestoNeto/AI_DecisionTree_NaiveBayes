import os
from Data import Data
from Models import Treino
from Visualization import Visualization

class Menu:
    def display_menu(self):
        print("Escolha o modelo para prever as emissões de CO2:")
        print("1. Árvore de Decisão")
        print("2. Naive Bayes")
        escolha = input("Digite o número do modelo desejado: ")
        
        file_path = r'C:\Users\modes\Documents\VScode\Python\IA\Trabalho_2\ConsumoCo2.csv' 
    
        if not os.path.exists(file_path):
            print(f"Arquivo não encontrado: {file_path}")
            return
        else:
            X_train, X_test, Y_train, Y_test = Data.loader(r'C:\Users\modes\Documents\VScode\Python\IA\Trabalho_2\ConsumoCo2.csv')
        
        if escolha == '1':
            Y_pred, mse = Treino.train_decision_tree(X_train, X_test, Y_train, Y_test)
            
        elif escolha == '2':
            Y_pred, mse = Treino.train_naive_bayes(X_train, X_test, Y_train, Y_test)
            
        Visualization.results(Y_test, Y_pred, escolha, mse)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

# Variáveis globais
video_path = None
resultados = []

# Função para mostrar dicas dos parâmetros 
def mostrar_dica_parametros():
    dicas_texto = (
        "Filtro Gaussiano:\n\n"
        "• Tamanho do Blur: Valores ímpares (3 a 25)\n"
        "  - Valores menores preservam mais detalhes\n"
        "  - Valores maiores suavizam mais a imagem\n\n"
        "• Limiar: Valores entre 15-60\n"
        "  - Valores baixos detectam mais movimento\n"
        "  - Valores altos reduzem falsos positivos\n\n"
        "• Área Mínima: Valores entre 300-1500\n"
        "  - Áreas pequenas detectam objetos menores\n"
        "  - Áreas grandes filtram ruídos e pequenos movimentos"
    )
    messagebox.showinfo("Dicas de Parâmetros", dicas_texto)


# Funções auxiliares
def selecionar_video(label_video):
    global video_path
    video_path = filedialog.askopenfilename(title="Selecione o Vídeo", 
                                          filetypes=[("Arquivos de vídeo", "*.mp4;*.avi;*.mov")])
    if video_path:
        label_video.config(text=f"Vídeo Selecionado:\n{video_path}", foreground="black")
    else:
        label_video.config(text="Nenhum vídeo selecionado", foreground="gray")

def iniciar_processamento(tamanho_suavizacao, limiar_binarizacao, area_min_contorno):
    global video_path, resultados
    
    if not video_path:
        messagebox.showerror("Erro", "Nenhum vídeo selecionado.")
        return

    try:
        movimento, resultado = processar_video(tamanho_suavizacao, limiar_binarizacao, area_min_contorno)

        if resultado is None:
            return

        # Anexar também os parâmetros ao resultado
        resultado['Parametros'] = f"Blur={tamanho_suavizacao}, Limiar={limiar_binarizacao}, Area={area_min_contorno}"
        resultados.append(resultado)

        exibir_grafico(movimento, len(resultados))  # passa o número do teste

    except ValueError:
        messagebox.showerror("Erro", "Erro ao processar os parâmetros. Verifique as seleções.")

def processar_video(tamanho_suavizacao, limiar_binarizacao, area_min_contorno):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo em '{video_path}'")
        exit()

    frame_anterior = None
    pixels_movimento_por_frame = []

    print("Pressione 'q' na janela do vídeo para sair.")
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Fim do vídeo")
            break

        # Pré-processamento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, tamanho_suavizacao, 0)

        if frame_anterior is None:
            frame_anterior = gray_blur
            continue

        # Calcula a diferença entre o frame anterior e o atual
        diff_frame = cv2.absdiff(frame_anterior, gray_blur)

        # Limiarização para criar a máscara de movimento
        _, thresh_frame = cv2.threshold(diff_frame, limiar_binarizacao, 255, cv2.THRESH_BINARY)
        
        frame_anterior = gray_blur
        
        # Pós-processamento da máscara
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # Conta os pixels brancos na máscara
        pixels_em_movimento = cv2.countNonZero(thresh_frame)
        pixels_movimento_por_frame.append(pixels_em_movimento)

        contornos, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            if cv2.contourArea(contorno) < area_min_contorno:
                continue

            (x, y, w, h) = cv2.boundingRect(contorno)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostra a máscara de movimento (resultado da limiarização)
        cv2.imshow("Mascara de Movimento", thresh_frame)
        cv2.imshow("Video Final com Deteccao", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Script finalizado. Gerando gráfico de movimento...")

    resultado = {
        'MediaPixels': np.mean(pixels_movimento_por_frame),
        'MaxPixels': np.max(pixels_movimento_por_frame)
    }

    return pixels_movimento_por_frame, resultado

def exibir_grafico(movimento, numero_teste):
    plt.figure(figsize=(12, 6))
    plt.plot(movimento)
    plt.title("Nível de Movimento por Frame")
    plt.xlabel("Número do Frame")
    plt.ylabel("Quantidade de Pixels em Movimento")
    plt.grid(True)

    # garantir diretório do arquivo .py
    diretorio = os.path.dirname(os.path.abspath(__file__))
    nome_arquivo = f"grafico_movimento_{numero_teste}.png"
    caminho = os.path.join(diretorio, nome_arquivo)
    plt.savefig(caminho)
    plt.show()
    messagebox.showinfo("Gráfico movimento", 
                          f"Grafico salvo como '{caminho}'")


def gerar_relatorio():
    global resultados
    
    if len(resultados) == 0:
        messagebox.showwarning("Atenção", "Nenhum teste foi realizado.")
        return

    df_resultados = pd.DataFrame(resultados)
    print("\nRelatório de Detecção de Movimento:")
    print(df_resultados)

    plt.figure(figsize=(12, 6))
    plt.bar(
        [r['Parametros'] for r in resultados],
        [r['MediaPixels'] for r in resultados]
    )
    plt.ylabel('Média de Pixels em Movimento')
    plt.title('Comparação dos Testes Realizados')
    plt.grid(False)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.show()

    try:
        diretorio = os.path.dirname(os.path.abspath(__file__))
        caminho_csv = os.path.join(diretorio, 'relatorio_deteccao_movimento.csv')
        df_resultados.to_csv(caminho_csv, index=False)
        messagebox.showinfo("Relatório Gerado", 
                          f"Relatório salvo como '{caminho_csv}'")
    except Exception as e:
        messagebox.showerror("Erro ao Salvar", f"Erro ao salvar relatório: {e}")


# Interface Tkinter
janela = tk.Tk()
janela.title("Detector de Movimento")
janela.geometry("450x300")

# Frame de configurações
frame_config = tk.LabelFrame(janela, text="Filtro Gaussiano", padx=10, pady=10)
frame_config.pack(pady=10)

# Seleção de vídeo
btn_selecionar_video = tk.Button(frame_config, text="Selecionar Vídeo", 
                                command=lambda: selecionar_video(label_video))
btn_selecionar_video.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")

label_video = tk.Label(frame_config, text="Nenhum vídeo selecionado", 
                       wraplength=550, justify="left", foreground="gray")
label_video.grid(row=1, column=0, columnspan=2, pady=5)

# Parâmetros
tk.Label(frame_config, text="Tamanho de Suavização:").grid(row=2, column=0, sticky="w")
combo_tamanho_suavizacao = ttk.Combobox(frame_config, values=['(3, 3)', '(5, 5)', '(7, 7)', '(9, 9)', '(11, 11)', '(15, 15)', '(25, 25)'])
combo_tamanho_suavizacao.grid(row=2, column=1, sticky="ew")
combo_tamanho_suavizacao.set('(15, 15)')

tk.Label(frame_config, text="Limiar de Binarização:").grid(row=3, column=0, sticky="w")
combo_limiar = ttk.Combobox(frame_config, values=['15', '20', '25', '30', '35', '40', '45', '50', '55', '60'])
combo_limiar.grid(row=3, column=1, sticky="ew")
combo_limiar.set('30')

tk.Label(frame_config, text="Área Mínima do Contorno:").grid(row=4, column=0, sticky="w")
combo_area = ttk.Combobox(frame_config, values=['300', '500', '700', '900', '1100', '1300', '1500'])
combo_area.grid(row=4, column=1, sticky="ew")
combo_area.set('700')

# Botão de dicas
btn_dicas = tk.Button(frame_config, text="?", width=3, 
                     command=mostrar_dica_parametros)
btn_dicas.grid(row=2, column=2, padx=5)

# Frame de ações
frame_acoes = tk.Frame(janela)
frame_acoes.pack(pady=15)

btn_iniciar = tk.Button(frame_acoes, text="Executar Teste", width=20,
                       command=lambda: iniciar_processamento(
                           eval(combo_tamanho_suavizacao.get()),
                           int(combo_limiar.get()),
                           int(combo_area.get())
                       ))
btn_iniciar.pack(side="left", padx=10)

btn_relatorio = tk.Button(frame_acoes, text="Gerar Relatório", width=20,
                         command=gerar_relatorio)
btn_relatorio.pack(side="left", padx=10)

janela.mainloop()

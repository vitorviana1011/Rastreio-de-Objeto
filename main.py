import cv2
import matplotlib.pyplot as plt

ARQUIVO_VIDEO = "video_transito2.mp4" 

TAMANHO_SUAVIZACAO = (15, 15) 
LIMIAR_BINARIZACAO = 30
AREA_MIN_CONTORNO = 700

cap = cv2.VideoCapture(ARQUIVO_VIDEO)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo em '{ARQUIVO_VIDEO}'")
    exit()

frame_anterior = None

#lista para guar o numero de pixels brancos
pixels_movimento_por_frame = []

print("Pressiona 'q' na janela do vídeo para sair.")
while True:
    ret, frame = cap.read()

    if not ret:
        print("Fim do vídeo, reiniciando...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_anterior = None 
        continue

    # Pré-processamento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, TAMANHO_SUAVIZACAO, 0)
    #tamanho_blur_mediana = 15 
    #gray_blur = cv2.medianBlur(gray, tamanho_blur_mediana)  
    
    if frame_anterior is None:
        frame_anterior = gray_blur
        continue

    #  Calcula a diferença entre o frame anterior e o atual
    diff_frame = cv2.absdiff(frame_anterior, gray_blur)

    # Limiarização para criar a máscara de movimento
    ret, thresh_frame = cv2.threshold(diff_frame, LIMIAR_BINARIZACAO, 255, cv2.THRESH_BINARY)
    
    frame_anterior = gray_blur
    
    # Pos-processamento da máscara
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    #conta os pixels brancos na mascara
    pixels_em_movimento = cv2.countNonZero(thresh_frame)
    pixels_movimento_por_frame.append(pixels_em_movimento)

    contornos, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        if cv2.contourArea(contorno) < AREA_MIN_CONTORNO:
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

plt.figure(figsize=(12, 6)) # Define o tamanho da figura do gráfico
plt.plot(pixels_movimento_por_frame) # Plota os dados que coletamos
plt.title("Nível de Movimento por Frame") # Título do gráfico
plt.xlabel("Número do Frame") # Rótulo do eixo X
plt.ylabel("Quantidade de Pixels em Movimento") # Rótulo do eixo Y
plt.grid(True) # Adiciona uma grade para melhor visualização
#plt.savefig("grafico_movimento.png") # Salva o gráfico como um arquivo de imagem
plt.show() # Exibe o gráfico em uma nova janela

print("Gráfico gerado e salvo como 'grafico_movimento.png'.")
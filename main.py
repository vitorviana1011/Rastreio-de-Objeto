import cv2

ARQUIVO_VIDEO = "video_transito2.mp4" 

TAMANHO_SUAVIZACAO = (15, 15) 
LIMIAR_BINARIZACAO = 30
AREA_MIN_CONTORNO = 500

cap = cv2.VideoCapture(ARQUIVO_VIDEO)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo em '{ARQUIVO_VIDEO}'")
    exit()

frame_anterior = None

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
print("Script finalizado.")
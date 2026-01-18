from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2 as cv
import numpy as np

def main():
    
    client = RemoteAPIClient()
    sim = client.require('sim')

    
    sim.setStepping(True)
    sim.startSimulation()

   
    sim.addLog(sim.verbosity_scriptinfos, f"Inicializando componentes del robot")
    right_motor = sim.getObject("/PioneerP3DX/rightMotor")
    left_motor = sim.getObject("/PioneerP3DX/leftMotor")
    camera = sim.getObject("/PioneerP3DX/camera")
    
    
    proximity_sensors = []
    for i in range(8):  # esto es para los primeros 8 sensores ultrasónicos
        try:
            sensor = sim.getObject(f"/PioneerP3DX/ultrasonicSensor[{i}]")
            proximity_sensors.append(sensor)
        except:
            break

   
    estado_robot = 0  
    posicion_x = 0
    posicion_y = 0

    # los motores al inicio estan en modo stop
    sim.setJointTargetVelocity(left_motor, 0)
    sim.setJointTargetVelocity(right_motor, 0)

    sim.addLog(sim.verbosity_scriptinfos, f"Iniciando ciclo principal de control")
    while True:
        
        colision_inminente = False
        for sensor in proximity_sensors:
            medicion = sim.readProximitySensor(sensor)
            if medicion and medicion[0] > 0 and medicion[1] < 0.20:  # hemos puesto un margen de seguridad 20cm
                colision_inminente = True
                break

        
        datos_imagen, dimensiones = sim.getVisionSensorImg(camera)
        imagen = np.frombuffer(datos_imagen, dtype=np.uint8)
        imagen = imagen.reshape([dimensiones[1], dimensiones[0], 3])
        imagen = cv.cvtColor(imagen, cv.COLOR_RGB2HSV)
        imagen = np.rot90(imagen, 2)
        imagen = np.fliplr(imagen)

        # los rangos de color para detectar el rojo
        rango_rojo_bajo_1 = np.array([0, 100, 100])
        rango_rojo_alto_1 = np.array([10, 255, 255])
        rango_rojo_bajo_2 = np.array([160, 100, 100])
        rango_rojo_alto_2 = np.array([180, 255, 255])

        
        mascara_1 = cv.inRange(imagen, rango_rojo_bajo_1, rango_rojo_alto_1)
        mascara_2 = cv.inRange(imagen, rango_rojo_bajo_2, rango_rojo_alto_2)
        mascara_combinada = cv.bitwise_or(mascara_1, mascara_2)

        momentos_mascara = cv.moments(mascara_combinada)
        objeto_visible = momentos_mascara["m00"] > 1000  #la area minima
        
        if objeto_visible:
            posicion_x = int(momentos_mascara["m10"] / momentos_mascara["m00"])
            posicion_y = int(momentos_mascara["m01"] / momentos_mascara["m00"])
            print(f"Detección confirmada - Coordenadas: {posicion_x}, {posicion_y}")

        
        imagen_monitoreo = mascara_combinada.copy()
        cv.namedWindow("Sistema de Visión", cv.WINDOW_NORMAL)
        cv.resizeWindow("Sistema de Visión", dimensiones[0], dimensiones[1])
        cv.imshow("Sistema de Visión", imagen_monitoreo)
        tecla_presionada = cv.waitKey(5)
        if tecla_presionada == 27:  
            break

        
        if colision_inminente:
           
            sim.setJointTargetVelocity(left_motor, 0)
            sim.setJointTargetVelocity(right_motor, 0)
            estado_robot = 2
            print("ALERTA: Peligro de colisión detectado!")
            
        elif estado_robot == 0:  #  explora
            if objeto_visible:
                estado_robot = 1  # busca
                print("Objetivo localizado! Activando modo persecución")
            else:
               
                sim.setJointTargetVelocity(left_motor, -0.5)
                sim.setJointTargetVelocity(right_motor, 0.5)
                
        elif estado_robot == 1:  #busca
            if not objeto_visible:
                estado_robot = 0  # vuelve a explorar
                print("Objetivo perdido! Reactivando búsqueda")
            else:
                
                centro_referencia = dimensiones[0] // 2
                if posicion_x < centro_referencia - 50:  #objetivo a la izquierda
                    sim.setJointTargetVelocity(left_motor, 0.3)
                    sim.setJointTargetVelocity(right_motor, 0.8)
                elif posicion_x > centro_referencia + 50:  #objetivo a la derecha
                    sim.setJointTargetVelocity(left_motor, 0.8)
                    sim.setJointTargetVelocity(right_motor, 0.3)
                else:  #objetivo centrado ya puede avanzar
                    sim.setJointTargetVelocity(left_motor, 0.8)
                    sim.setJointTargetVelocity(right_motor, 0.8)
                    
        elif estado_robot == 2: 
           
            if not colision_inminente:
                estado_robot = 0
                print("Seguimos con la búsqueda")

        sim.step()

    # Procedimiento de finalización
    sim.setJointTargetVelocity(left_motor, 0)
    sim.setJointTargetVelocity(right_motor, 0)
    cv.destroyAllWindows()
    sim.stopSimulation()

if __name__ == "__main__":
    main()
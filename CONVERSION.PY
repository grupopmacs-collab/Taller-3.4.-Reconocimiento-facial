import tensorflow as tf
import numpy as np
import os
import json
import time
from tensorflow.keras.models import load_model

class TFLiteConverter:
    def __init__(self, model_path, metadata_path):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.class_names = []

    def load_model_and_metadata(self):
        """Carga el modelo y metadatos"""
        print("=== CARGA PARA CONVERSIÓN TFLITE ===")

        try:
            self.model = load_model(self.model_path)
            print(f"✅ Modelo cargado: {self.model_path}")

            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            self.class_names = metadata['class_names']
            self.img_size = tuple(metadata['img_size'])

            print(f"✅ Clases: {self.class_names}")
            print(f"✅ Tamaño de imagen: {self.img_size}")

            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False

    def convert_to_tflite(self, optimize=True, quantize=False):
        """Convierte el modelo a TensorFlow Lite"""
        print("\n=== CONVERSIÓN A TENSORFLOW LITE ===")

        if self.model is None:
            print("❌ Primero debe cargar el modelo")
            return None

        # Crear función de conversión para TFLite
        def representative_data_gen():
            for _ in range(100):
                yield [np.random.uniform(0, 1, size=(1, *self.img_size, 3)).astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Opciones de optimización
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("✅ Optimizaciones DEFAULT aplicadas")

        if quantize:
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            print("✅ Cuantización INT8 aplicada")

        # Convertir modelo
        try:
            tflite_model = converter.convert()
            print("✅ Conversión a TFLite exitosa")

            # Guardar modelo TFLite
            tflite_filename = self.model_path.replace('.h5', '.tflite')
            with open(tflite_filename, 'wb') as f:
                f.write(tflite_model)

            # Tamaño del modelo
            original_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
            tflite_size = os.path.getsize(tflite_filename) / (1024 * 1024)    # MB

            print(f"📊 Tamaño del modelo original: {original_size:.2f} MB")
            print(f"📊 Tamaño del modelo TFLite: {tflite_size:.2f} MB")
            print(f"📉 Reducción: {((original_size - tflite_size) / original_size * 100):.1f}%")

            return tflite_filename

        except Exception as e:
            print(f"❌ Error en conversión: {e}")
            return None

    def test_tflite_model(self, tflite_path, test_samples=10):
        """Prueba el modelo TFLite con datos de ejemplo"""
        print("\n=== PRUEBA DEL MODELO TFLITE ===")

        # Cargar el modelo TFLite
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Obtener detalles de entrada/salida
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"✅ Input details: {input_details[0]['shape']}")
        print(f"✅ Output details: {output_details[0]['shape']}")

        # Probar con datos aleatorios
        latencies = []
        for i in range(test_samples):
            # Crear datos de entrada
            input_data = np.random.uniform(0, 1, size=(1, *self.img_size, 3)).astype(np.float32)

            # Medir tiempo de inferencia
            start_time = time.time()

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            inference_time = (time.time() - start_time) * 1000  # ms

            latencies.append(inference_time)

            if i < 3:  # Mostrar primeras predicciones
                predicted_class = np.argmax(output_data[0])
                confidence = np.max(output_data[0])
                print(f"   Muestra {i+1}: Clase '{self.class_names[predicted_class]}' - Confianza: {confidence:.4f} - Tiempo: {inference_time:.2f}ms")

        # Estadísticas de latencia
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)

        print(f"\n📊 ESTADÍSTICAS DE LATENCIA:")
        print(f"   ⏱️  Latencia promedio: {avg_latency:.2f} ms")
        print(f"   ⏱️  Latencia mínima: {min_latency:.2f} ms")
        print(f"   ⏱️  Latencia máxima: {max_latency:.2f} ms")

        # Verificar criterio de rendimiento
        if avg_latency < 500:
            print(f"✅ CUMPLE CRITERIO: Latencia < 500 ms (actual: {avg_latency:.2f} ms)")
        else:
            print(f"❌ NO CUMPLE CRITERIO: Latencia > 500 ms (actual: {avg_latency:.2f} ms)")

        return {
            'avg_latency': avg_latency,
            'min_latency': min_latency,
            'max_latency': max_latency,
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': output_details
        }

    def save_tflite_metadata(self, tflite_path):
        """Guarda metadatos para la aplicación móvil"""
        print("\n=== GUARDANDO METADATOS PARA APLICACIÓN MÓVIL ===")

        metadata = {
            'class_names': self.class_names,
            'img_size': self.img_size,
            'model_type': 'tflite',
            'conversion_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'input_shape': [1, *self.img_size, 3],
            'output_shape': [1, len(self.class_names)]
        }

        metadata_filename = tflite_path.replace('.tflite', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadatos guardados: {metadata_filename}")

        # Crear archivo de labels para la app
        labels_filename = tflite_path.replace('.tflite', '_labels.txt')
        with open(labels_filename, 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")

        print(f"✅ Archivo de labels guardado: {labels_filename}")

        return metadata_filename

def ejecutar_conversion_tflite():
    """Función principal para conversión TFLite"""
    print("🎯 INICIANDO CONVERSIÓN TENSORFLOW LITE")
    print("=" * 60)

    # Buscar archivos más recientes
    model_files = [f for f in os.listdir('.') if f.startswith('facial_recognition_model_') and f.endswith('.h5')]
    metadata_files = [f for f in os.listdir('.') if f.startswith('model_metadata_') and f.endswith('.json')]

    if not model_files or not metadata_files:
        print("❌ No se encontraron archivos de modelo")
        return

    latest_model = sorted(model_files)[-1]
    latest_metadata = sorted(metadata_files)[-1]

    print(f"📁 Modelo: {latest_model}")
    print(f"📁 Metadatos: {latest_metadata}")

    # Ejecutar conversión
    converter = TFLiteConverter(latest_model, latest_metadata)

    if converter.load_model_and_metadata():
        # Conversión optimizada (recomendada para móviles)
        tflite_path = converter.convert_to_tflite(optimize=True, quantize=False)

        if tflite_path:
            # Probar modelo convertido
            test_results = converter.test_tflite_model(tflite_path)

            # Guardar metadatos para la app
            metadata_path = converter.save_tflite_metadata(tflite_path)

            print(f"\n✅ CONVERSIÓN COMPLETADA EXITOSAMENTE!")
            print(f"📁 Modelo TFLite: {tflite_path}")
            print(f"📁 Metadatos: {metadata_path}")

            return {
                'tflite_path': tflite_path,
                'metadata_path': metadata_path,
                'test_results': test_results
            }

    return None

# EJECUTAR CONVERSIÓN
if __name__ == "__main__":
    conversion_results = ejecutar_conversion_tflite()
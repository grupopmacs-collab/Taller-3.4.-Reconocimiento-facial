import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

class FacialRecognitionModel:
    def __init__(self, datasets_path, num_classes, img_size=(224, 224)):
        self.datasets_path = datasets_path
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None
        self.class_names = []

    def setup_data_generators(self, batch_size=32):
        """Configura los generadores de datos para entrenamiento, validación y prueba"""
        print("=== CONFIGURACIÓN DE GENERADORES DE DATOS ===")

        # Data augmentation solo para entrenamiento
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )

        # Solo reescalar para validación y prueba
        test_val_datagen = ImageDataGenerator(rescale=1./255)

        # Generadores de datos
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.datasets_path, 'train'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

        self.validation_generator = test_val_datagen.flow_from_directory(
            os.path.join(self.datasets_path, 'validation'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.test_generator = test_val_datagen.flow_from_directory(
            os.path.join(self.datasets_path, 'test'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Obtener nombres de las clases
        self.class_names = list(self.train_generator.class_indices.keys())

        print(f"✅ Clases detectadas: {self.class_names}")
        print(f"📊 Imágenes de entrenamiento: {self.train_generator.samples}")
        print(f"📊 Imágenes de validación: {self.validation_generator.samples}")
        print(f"📊 Imágenes de prueba: {self.test_generator.samples}")

        return self.train_generator, self.validation_generator, self.test_generator

    def create_model(self, learning_rate=0.001, dropout_rate=0.3):
        """Crea el modelo usando Transfer Learning con MobileNetV2"""
        print("\n=== CREACIÓN DEL MODELO ===")
        print("🔄 Cargando MobileNetV2 pre-entrenado...")

        # Cargar MobileNetV2 pre-entrenado (sin la capa superior)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )

        # Congelar las capas base (opcional - podemos descongelar después)
        base_model.trainable = False

        print(f"✅ Base Model: MobileNetV2")
        print(f"✅ Capas congeladas: {len(base_model.layers)}")

        # Construir el modelo personalizado
        inputs = tf.keras.Input(shape=(*self.img_size, 3))

        # Capa base (pre-entrenada)
        x = base_model(inputs, training=False)

        # Capas personalizadas
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)

        # Capa de salida
        outputs = Dense(self.num_classes, activation='softmax')(x)

        # Modelo completo
        self.model = Model(inputs, outputs)

        # Compilar el modelo
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        print("✅ Modelo compilado exitosamente!")
        print(f"📈 Learning Rate: {learning_rate}")
        print(f"🎯 Función de pérdida: Categorical Cross-Entropy")
        print(f"⚙️  Optimizador: Adam")

        return self.model

    def unfreeze_layers(self, unfreeze_after=100):
        """Descongela algunas capas del modelo base para fine-tuning"""
        print(f"\n=== FINE-TUNING - DESCONGELANDO CAPAS ===")

        if self.model is None:
            print("❌ Primero debe crear el modelo")
            return

        # Hacer el base_model entrenable
        base_model = self.model.layers[1]  # MobileNetV2 es la segunda capa
        base_model.trainable = True

        # Descongelar solo las últimas capas
        for layer in base_model.layers[:unfreeze_after]:
            layer.trainable = False

        # Recompilar con learning rate más bajo para fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),  # LR más bajo para fine-tuning
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        trainable_count = sum([w.shape.num_elements() for w in self.model.trainable_weights])
        non_trainable_count = sum([w.shape.num_elements() for w in self.model.non_trainable_weights])

        print(f"✅ Capas descongeladas: {unfreeze_after}+")
        print(f"📊 Parámetros entrenables: {trainable_count:,}")
        print(f"📊 Parámetros no entrenables: {non_trainable_count:,}")
        print(f"📈 Nuevo Learning Rate: 0.0001")

    def setup_callbacks(self):
        """Configura los callbacks para el entrenamiento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        callbacks = [
            # Early Stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce Learning Rate on Plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),

            # Model Checkpoint
            ModelCheckpoint(
                f'best_model_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),

            # TensorBoard (opcional)
            # tf.keras.callbacks.TensorBoard(log_dir=f'./logs_{timestamp}')
        ]

        return callbacks

    def train_model(self, epochs=50, initial_epochs=20):
        """Entrena el modelo en dos fases: con capas congeladas y fine-tuning"""
        print("\n=== ENTRENAMIENTO DEL MODELO ===")

        if self.model is None:
            print("❌ Primero debe crear el modelo")
            return

        # Fase 1: Entrenamiento con capas base congeladas
        print("🎯 FASE 1: Entrenamiento con capas base congeladas")

        callbacks = self.setup_callbacks()

        history_phase1 = self.model.fit(
            self.train_generator,
            epochs=initial_epochs,
            validation_data=self.validation_generator,
            callbacks=callbacks,
            verbose=1
        )

        # Fase 2: Fine-tuning con capas descongeladas
        print("\n🎯 FASE 2: Fine-tuning con capas descongeladas")
        self.unfreeze_layers(unfreeze_after=100)

        # Callbacks para fine-tuning
        fine_tune_callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]

        history_phase2 = self.model.fit(
            self.train_generator,
            initial_epoch=initial_epochs,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=fine_tune_callbacks,
            verbose=1
        )

        # Combinar historiales
        self.history = {
            'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
            'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
            'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
            'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
        }

        print("✅ Entrenamiento completado!")

        return self.history

    def evaluate_model(self):
        """Evalúa el modelo en el conjunto de prueba"""
        print("\n=== EVALUACIÓN EN CONJUNTO DE PRUEBA ===")

        if self.model is None:
            print("❌ Primero debe entrenar el modelo")
            return

        # Evaluar en test set
        test_results = self.model.evaluate(self.test_generator, verbose=0)

        print("📊 RESULTADOS EN TEST SET:")
        print(f"   ✅ Loss: {test_results[0]:.4f}")
        print(f"   ✅ Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
        print(f"   ✅ Precision: {test_results[2]:.4f}")
        print(f"   ✅ Recall: {test_results[3]:.4f}")

        # Calcular F1-Score
        f1 = 2 * (test_results[2] * test_results[3]) / (test_results[2] + test_results[3])
        print(f"   ✅ F1-Score: {f1:.4f}")

        # Verificar criterio de éxito
        accuracy = test_results[1]
        if accuracy >= 0.90:
            print(f"🎉 ¡OBJETIVO CUMPLIDO! Accuracy >= 90%: {accuracy*100:.2f}%")
        else:
            print(f"📝 Objetivo no alcanzado. Accuracy actual: {accuracy*100:.2f}%")

        return test_results

    def plot_training_history(self):
        """Grafica el historial de entrenamiento"""
        if self.history is None:
            print("❌ No hay historial de entrenamiento para graficar")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Gráfico de Accuracy
        axes[0].plot(self.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Accuracy durante el Entrenamiento', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Gráfico de Loss
        axes[1].plot(self.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Loss durante el Entrenamiento', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_model(self, model_name="facial_recognition_model"):
        """Guarda el modelo y sus metadatos"""
        print(f"\n=== GUARDANDO MODELO ===")

        if self.model is None:
            print("❌ No hay modelo para guardar")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.h5"

        # Guardar modelo
        self.model.save(model_filename)

        # Guardar metadatos
        metadata = {
            'class_names': self.class_names,
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'training_date': timestamp,
            'test_accuracy': self.history['val_accuracy'][-1] if self.history else None
        }

        with open(f'model_metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Modelo guardado como: {model_filename}")
        print(f"✅ Metadatos guardados: model_metadata_{timestamp}.json")

        return model_filename

def ejecutar_fase_4():
    """Función principal para ejecutar la Fase 4"""
    print("🎯 INICIANDO FASE 4: MODELADO")
    print("=" * 60)

    # Configurar rutas
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = '/content/drive/MyDrive/datasets_finales'
        print("✅ Google Drive montado correctamente")
    except:
        print("⚠️  Ejecutando en entorno local")
        base_path = './datasets_finales'

    # Verificar que existe el dataset procesado
    if not os.path.exists(base_path):
        print(f"❌ No se encuentra el dataset procesado en: {base_path}")
        print("💡 Ejecuta primero la Fase 3: Preparación de Datos")
        return

    # Contar número de clases
    train_path = os.path.join(base_path, 'train')
    num_classes = len([name for name in os.listdir(train_path)
                      if os.path.isdir(os.path.join(train_path, name))])

    print(f"🔍 Detectadas {num_classes} clases en el dataset")

    # Crear y entrenar modelo
    model_builder = FacialRecognitionModel(
        datasets_path=base_path,
        num_classes=num_classes,
        img_size=(224, 224)
    )

    # Configurar generadores de datos
    model_builder.setup_data_generators(batch_size=32)

    # Crear modelo
    model_builder.create_model(learning_rate=0.001, dropout_rate=0.3)

    # Mostrar arquitectura del modelo
    print("\n=== ARQUITECTURA DEL MODELO ===")
    model_builder.model.summary()

    # Entrenar modelo
    history = model_builder.train_model(epochs=50, initial_epochs=20)

    # Evaluar modelo
    test_results = model_builder.evaluate_model()

    # Visualizar entrenamiento
    model_builder.plot_training_history()

    # Guardar modelo
    saved_model_path = model_builder.save_model()

    print("\n✅ FASE 4 COMPLETADA EXITOSAMENTE!")
    print(f"📁 Modelo guardado en: {saved_model_path}")

    # Recomendaciones basadas en resultados
    accuracy = test_results[1]
    if accuracy >= 0.90:
        print("🎉 ¡Excelente! El modelo cumple con el objetivo del 90% de accuracy")
        print("📱 Puedes proceder con la conversión a TensorFlow Lite")
    elif accuracy >= 0.80:
        print("📈 Buen rendimiento, considera:")
        print("   - Aumentar el número de épocas")
        print("   - Probar diferentes arquitecturas")
        print("   - Recolectar más datos")
    else:
        print("🔄 Se recomienda mejorar el modelo:")
        print("   - Revisar la calidad de los datos")
        print("   - Ajustar hiperparámetros")
        print("   - Probar data augmentation más agresivo")

    return model_builder

# EJECUTAR FASE 4
if __name__ == "__main__":
    # Verificar si TensorFlow puede usar GPU
    print("🔍 Verificando disponibilidad de GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU disponible: {gpus[0]}")
        # Configurar memoria GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  No se detectó GPU, usando CPU")

    # Ejecutar fase 4
    model_builder = ejecutar_fase_4()
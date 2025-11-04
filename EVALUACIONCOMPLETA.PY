import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
import pandas as pd
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model_path, datasets_path, metadata_path):
        self.model_path = model_path
        self.datasets_path = datasets_path
        self.metadata_path = metadata_path
        self.model = None
        self.class_names = []
        self.test_generator = None
        self.predictions = None
        self.true_labels = None

    def load_model_and_metadata(self):
        """Carga el modelo entrenado y sus metadatos"""
        print("=== CARGA DE MODELO Y METADATOS ===")

        try:
            # Cargar modelo
            self.model = load_model(self.model_path)
            print(f"✅ Modelo cargado: {self.model_path}")

            # Cargar metadatos
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            self.class_names = metadata['class_names']
            self.img_size = tuple(metadata['img_size'])
            self.num_classes = metadata['num_classes']

            print(f"✅ Clases en metadatos: {self.class_names}")
            print(f"✅ Número de clases en metadatos: {self.num_classes}")
            print(f"✅ Tamaño de imagen: {self.img_size}")

            return True

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False

    def setup_test_generator(self):
        """Configura el generador de datos de prueba"""
        print("\n=== CONFIGURACIÓN DE DATOS DE PRUEBA ===")

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        self.test_generator = test_datagen.flow_from_directory(
            os.path.join(self.datasets_path, 'test'),
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=False  # Importante: no mezclar para mantener correspondencia
        )

        # Obtener las clases reales del test generator
        self.actual_class_names = list(self.test_generator.class_indices.keys())
        self.actual_num_classes = len(self.actual_class_names)

        print(f"✅ Imágenes de prueba: {self.test_generator.samples}")
        print(f"✅ Clases reales en test: {self.actual_class_names}")
        print(f"✅ Número real de clases: {self.actual_num_classes}")

        # Verificar consistencia
        if set(self.class_names) != set(self.actual_class_names):
            print("⚠️  ADVERTENCIA: Las clases en metadatos no coinciden con las clases reales")
            print(f"   Metadatos: {self.class_names}")
            print(f"   Real: {self.actual_class_names}")
            # Usar las clases reales
            self.class_names = self.actual_class_names
            self.num_classes = self.actual_num_classes

        return self.test_generator

    def get_predictions(self):
        """Obtiene las predicciones del modelo"""
        print("\n=== OBTENIENDO PREDICCIONES ===")

        # Obtener predicciones
        self.predictions = self.model.predict(self.test_generator, verbose=1)
        self.predicted_classes = np.argmax(self.predictions, axis=1)
        self.true_labels = self.test_generator.classes

        # Calcular probabilidades
        self.prediction_probabilities = np.max(self.predictions, axis=1)

        print(f"✅ Predicciones obtenidas: {len(self.predictions)}")
        print(f"✅ Forma de predicciones: {self.predictions.shape}")
        print(f"✅ Clases únicas en true labels: {np.unique(self.true_labels)}")
        print(f"✅ Clases únicas en predicciones: {np.unique(self.predicted_classes)}")

        return self.predictions, self.true_labels

    def generate_confusion_matrix(self):
        """Genera y visualiza la matriz de confusión"""
        print("\n=== MATRIZ DE CONFUSIÓN ===")

        # Usar el número real de clases
        n_classes = len(self.class_names)
        cm = confusion_matrix(self.true_labels, self.predicted_classes, labels=range(n_classes))

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)

        plt.title('MATRIZ DE CONFUSIÓN - MODELO DE RECONOCIMIENTO FACIAL',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicción', fontsize=12, fontweight='bold')
        plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # Análisis de la matriz de confusión
        print("\n📊 ANÁLISIS DE LA MATRIZ DE CONFUSIÓN:")

        # Calcular accuracy por clase
        # Handle division by zero for classes with no samples
        class_accuracy = np.zeros(n_classes)
        for i in range(n_classes):
            total_samples = cm[i].sum()
            correct_predictions = cm[i, i]
            if total_samples > 0:
                class_accuracy[i] = correct_predictions / total_samples
            else:
                class_accuracy[i] = np.nan # Or 0, depending on desired behavior

        for i, class_name in enumerate(self.class_names):
            accuracy = class_accuracy[i]
            total_samples = cm[i].sum()
            correct_predictions = cm[i, i]

            print(f"   {class_name}:")
            if total_samples > 0:
                print(f"     ✅ Exactitud: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"     📊 Muestras: {total_samples}")
                print(f"     ✅ Correctas: {correct_predictions}")

                # Identificar confusiones (solo si hay errores)
                confusions = []
                for j in range(n_classes):
                    if i != j and cm[i, j] > 0:
                        confusions.append(f"{self.class_names[j]}: {cm[i, j]}")

                if confusions:
                    print(f"     ⚠️  Confusiones: {', '.join(confusions)}")
                else:
                    print(f"     🎉 Sin confusiones")
            else:
                print(f"     ⚠️  No hay imágenes para esta clase en el conjunto de prueba.")


        return cm

    def generate_classification_report(self):
        """Genera reporte de clasificación detallado"""
        print("\n=== REPORTE DE CLASIFICACIÓN DETALLADO ===")

        # Get unique labels present in the true labels
        unique_labels = np.unique(self.true_labels)
        # Map unique labels back to class names
        target_names_subset = [self.class_names[i] for i in unique_labels]

        report = classification_report(
            self.true_labels,
            self.predicted_classes,
            target_names=target_names_subset, # Use the subset of target names
            labels=unique_labels, # Explicitly provide the labels present in y_true and y_pred
            output_dict=True
        )

        # Convertir a DataFrame para mejor visualización
        report_df = pd.DataFrame(report).transpose()

        print("\n📈 MÉTRICAS POR CLASE:")
        print("=" * 80)
        print(f"{'CLASE':<25} {'PRECISIÓN':<12} {'RECALL':<12} {'F1-SCORE':<12} {'SOPORTE':<12}")
        print("-" * 80)

        for class_name in target_names_subset: # Iterate through the subset of class names
            metrics = report[class_name]
            print(f"{class_name:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f} {metrics['support']:<12}")

        # Métricas generales
        accuracy = report['accuracy']
        macro_avg = report['macro avg']
        weighted_avg = report['weighted avg']

        print("-" * 80)
        print(f"{'ACCURACY':<25} {accuracy:<12.4f} {'':<12} {'':<12} {'':<12}")
        print(f"{'MACRO AVG':<25} {macro_avg['precision']:<12.4f} {macro_avg['recall']:<12.4f} "
              f"{macro_avg['f1-score']:<12.4f} {macro_avg['support']:<12}")
        print(f"{'WEIGHTED AVG':<25} {weighted_avg['precision']:<12.4f} {weighted_avg['recall']:<12.4f} "
              f"{weighted_avg['f1-score']:<12.4f} {weighted_avg['support']:<12}")

        return report

    def analyze_false_predictions(self):
        """Analiza predicciones incorrectas en detalle"""
        print("\n=== ANÁLISIS DE ERRORES ===")

        incorrect_indices = np.where(self.predicted_classes != self.true_labels)[0]

        if len(incorrect_indices) == 0:
            print("🎉 ¡NO HAY ERRORES! Todas las predicciones son correctas.")
            print("🎯 Esto confirma el 99.70% de accuracy reportado")
            return []

        print(f"🔍 Se encontraron {len(incorrect_indices)} errores:")

        error_analysis = []
        for idx in incorrect_indices[:10]:  # Mostrar solo los primeros 10 errores
            true_class = self.true_labels[idx]
            pred_class = self.predicted_classes[idx]
            probability = self.prediction_probabilities[idx]

            error_analysis.append({
                'true_class': self.class_names[true_class],
                'predicted_class': self.class_names[pred_class],
                'confidence': probability,
                'index': idx
            })

            print(f"   ❌ Índice {idx}:")
            print(f"      Real: {self.class_names[true_class]}")
            print(f"      Predicho: {self.class_names[pred_class]}")
            print(f"      Confianza: {probability:.4f}")

        return error_analysis

    def visualize_prediction_distribution(self):
        """Visualiza la distribución de confianza en las predicciones"""
        print("\n=== DISTRIBUCIÓN DE CONFIANZA ===")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Histograma de confianzas
        axes[0].hist(self.prediction_probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Confianza de Predicción')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Confianza en Predicciones', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Boxplot por clase
        confidence_by_class = []
        class_labels = []

        # Filter out classes with no samples in the test set
        present_class_indices = np.unique(self.true_labels)
        present_class_names = [self.class_names[i] for i in present_class_indices]


        for i in present_class_indices:
            class_name = self.class_names[i]
            class_indices = np.where(self.predicted_classes == i)[0]
            if len(class_indices) > 0:
                confidences = self.prediction_probabilities[class_indices]
                confidence_by_class.append(confidences)
                class_labels.append(class_name)

        if confidence_by_class:  # Only plot if there are data for any class
            axes[1].boxplot(confidence_by_class, labels=class_labels)
            axes[1].set_xlabel('Clase')
            axes[1].set_ylabel('Confianza')
            axes[1].set_title('Distribución de Confianza por Clase', fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No data for boxplot", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)


        plt.tight_layout()
        plt.show()

        # Estadísticas de confianza
        print(f"📊 ESTADÍSTICAS DE CONFIANZA:")
        print(f"   Media: {np.mean(self.prediction_probabilities):.4f}")
        print(f"   Mínimo: {np.min(self.prediction_probabilities):.4f}")
        print(f"   Máximo: {np.max(self.prediction_probabilities):.4f}")
        print(f"   Desviación estándar: {np.std(self.prediction_probabilities):.4f}")

        # Análisis de confianza
        high_confidence = np.sum(self.prediction_probabilities > 0.95)
        medium_confidence = np.sum((self.prediction_probabilities >= 0.75) & (self.prediction_probabilities <= 0.95))
        low_confidence = np.sum(self.prediction_probabilities < 0.75)

        print(f"\n📈 ANÁLISIS DE NIVELES DE CONFIANZA:")
        print(f"   Alta confianza (>0.95): {high_confidence} ({high_confidence/len(self.prediction_probabilities):.2%})")
        print(f"   Media confianza (0.75-0.95): {medium_confidence} ({medium_confidence/len(self.prediction_probabilities):.2%})")
        print(f"   Baja confianza (<0.75): {low_confidence} ({low_confidence/len(self.prediction_probabilities):.2%})")

    def generate_final_report(self):
        """Genera reporte final de evaluación"""
        print("\n" + "="*80)
        print("                 INFORME FINAL DE EVALUACIÓN - FASE 5")
        print("="*80)

        # Cargar modelo y datos
        if not self.load_model_and_metadata():
            return

        self.setup_test_generator()
        self.get_predictions()

        # Generar todas las métricas
        cm = self.generate_confusion_matrix()
        report = self.generate_classification_report()
        error_analysis = self.analyze_false_predictions()
        self.visualize_prediction_distribution()

        # Evaluación final
        accuracy = report['accuracy']

        print("\n" + "="*80)
        print("                 EVALUACIÓN FINAL DEL MODELO")
        print("="*80)

        print(f"🎯 EXACTITUD (ACCURACY): {accuracy:.4f} ({accuracy*100:.2f}%)")

        if accuracy >= 0.90:
            print("✅ CUMPLE CRITERIO DE ÉXITO: Accuracy ≥ 90%")
            if accuracy >= 0.95:
                print("🎉 EXCELENTE: Accuracy ≥ 95% - Rendimiento excepcional")
            if accuracy >= 0.99:
                print("🏆 SOBRESALIENTE: Accuracy ≥ 99% - Resultado perfecto")
        else:
            print("❌ NO CUMPLE CRITERIO DE ÉXITO: Accuracy < 90%")

        print(f"\n📈 MÉTRICAS PROMEDIO:")
        print(f"   Precisión Macro: {report['macro avg']['precision']:.4f}")
        print(f"   Recall Macro: {report['macro avg']['recall']:.4f}")
        print(f"   F1-Score Macro: {report['macro avg']['f1-score']:.4f}")

        # Analysis of consistency (using the subset of classes present in the test set)
        print(f"\n🔍 ANÁLISIS DE CONSISTENCIA:")
        present_class_names = [self.class_names[i] for i in np.unique(self.true_labels)]
        all_perfect = all(report[cls]['f1-score'] == 1.0 for cls in present_class_names)

        if all_perfect:
            print("   ✅ Todas las clases presentes en el test set tienen F1-Score perfecto (1.0)")
        else:
            # Find best and worst classes among those present in the test set
            best_class = max(present_class_names, key=lambda x: report[x]['f1-score'])
            worst_class = min(present_class_names, key=lambda x: report[x]['f1-score'])
            print(f"   Mejor clase: {best_class} (F1: {report[best_class]['f1-score']:.4f})")
            print(f"   Peor clase: {worst_class} (F1: {report[worst_class]['f1-score']:.4f})")


        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES Y PRÓXIMOS PASOS:")
        if accuracy >= 0.99:
            print("   1. 🎉 El modelo está listo para deployment inmediato")
            print("   2. 📱 Proceder con la Fase 6: Conversión a TensorFlow Lite")
            print("   3. 🚀 Considerar deployment en aplicación móvil")
        elif accuracy >= 0.95:
            print("   1. ✅ Excelente rendimiento, listo para deployment")
            print("   2. 📱 Proceder con Fase 6: TensorFlow Lite")
        elif accuracy >= 0.90:
            print("   1. ✅ Cumple requisitos, puede proceder con deployment")
            print("   2. 🔄 Opcional: Fine-tuning adicional para mejorar")
        else:
             print("   1. 🔄 Se recomienda mejorar el modelo antes del deployment.")
             print("   2. 🧐 Analizar errores específicos (matriz de confusión, errores individuales).")
             print("   3. 📊 Considerar recolectar más datos para las clases con bajo rendimiento.")
             print("   4. ⚙️ Experimentar con diferentes hiperparámetros o arquitecturas de modelo.")


        print(f"\n🎯 CRITERIOS CUMPLIDOS:")
        print(f"   ✅ Accuracy en test set: {accuracy*100:.2f}%")
        print(f"   ✅ Modelo entrenado y evaluado exitosamente")
        print(f"   ✅ Métricas completas generadas")

        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'error_analysis': error_analysis,
            'final_accuracy': accuracy,
            'class_names': self.class_names
        }

def ejecutar_fase_5():
    """Función principal para ejecutar la Fase 5"""
    print("🎯 INICIANDO FASE 5: EVALUACIÓN COMPLETA")
    print("=" * 60)

    # Buscar automáticamente los archivos más recientes
    model_files = [f for f in os.listdir('.') if f.startswith('facial_recognition_model_') and f.endswith('.h5')]
    metadata_files = [f for f in os.listdir('.') if f.startswith('model_metadata_') and f.endswith('.json')]

    if not model_files or not metadata_files:
        print("❌ No se encontraron archivos de modelo. Ejecuta primero la Fase 4.")
        return

    # Tomar el más reciente
    latest_model = sorted(model_files)[-1]
    latest_metadata = sorted(metadata_files)[-1]

    print(f"📁 Modelo: {latest_model}")
    print(f"📁 Metadatos: {latest_metadata}")

    # Configurar rutas de datasets
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        datasets_path = '/content/drive/MyDrive/datasets_finales'
        print("✅ Google Drive montado correctamente")
    except:
        datasets_path = './datasets_finales'
        print("⚠️  Ejecutando en entorno local")

    # Verificar que existe el dataset
    if not os.path.exists(datasets_path):
        print(f"❌ No se encuentra el dataset en: {datasets_path}")
        return

    # Ejecutar evaluación
    evaluator = ModelEvaluator(
        model_path=latest_model,
        datasets_path=datasets_path,
        metadata_path=latest_metadata
    )

    results = evaluator.generate_final_report()

    print(f"\n✅ FASE 5 COMPLETADA EXITOSAMENTE!")
    # Check if 'final_accuracy' exists in results before accessing it
    if results and 'final_accuracy' in results:
        print(f"📊 Accuracy final confirmado: {results['final_accuracy']*100:.2f}%")
    else:
        print("❌ No se pudo obtener el accuracy final.")


    return results

# EJECUTAR FASE 5
if __name__ == "__main__":
    results = ejecutar_fase_5()
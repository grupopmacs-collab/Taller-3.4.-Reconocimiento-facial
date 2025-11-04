import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreparer:
    def __init__(self, dataset_path, output_size=(224, 224)):
        self.dataset_path = dataset_path
        self.output_size = output_size
        self.preprocessed_path = os.path.join(os.path.dirname(dataset_path), 'dataset_preprocesado')
        self.augmented_path = os.path.join(os.path.dirname(dataset_path), 'dataset_aumentado')
        self.estadisticas_preprocesamiento = {}

        # Lista oficial de integrantes
        self.integrantes_oficiales = [
            "Lizarraga_Aliaga_Alexis_Joaquin",
            "Huaychi_Sencia_Jose_Natividad",
            "Challco_Ccohuanqui_Samir_Junior",
            "Tejada_Taco_Francys_Raul",
            "Zavala_Huacarpuma_Juan_Aldair"
        ]

    def verificar_fase_2(self):
        """Verifica que la Fase 2 se haya completado correctamente"""
        print("=== VERIFICACIÓN FASE 2 ===")

        if not os.path.exists(self.dataset_path):
            print(f"❌ No se encuentra el dataset: {self.dataset_path}")
            return False

        integrantes_con_imagenes = []
        for integrante in self.integrantes_oficiales:
            integrante_path = os.path.join(self.dataset_path, integrante)
            if os.path.exists(integrante_path):
                imagenes = [f for f in os.listdir(integrante_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imagenes:
                    integrantes_con_imagenes.append(integrante)

        print(f"✅ Integrantes con imágenes: {len(integrantes_con_imagenes)}/5")
        for integrante in integrantes_con_imagenes:
            print(f"   - {integrante}")

        if len(integrantes_con_imagenes) < 3:
            print("❌ Se necesitan al menos 3 integrantes con imágenes")
            return False

        return True

    def detectar_y_recortar_rostros(self):
        """Detección y recorte facial usando OpenCV"""
        print("\n=== DETECCIÓN Y RECORTE FACIAL ===")

        # Cargar detector de rostros
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        rostros_detectados = 0
        rostros_no_detectados = 0

        # Crear directorio de salida
        if os.path.exists(self.preprocessed_path):
            shutil.rmtree(self.preprocessed_path)
        os.makedirs(self.preprocessed_path)

        for integrante in self.integrantes_oficiales:
            integrante_path = os.path.join(self.dataset_path, integrante)
            output_integrante_path = os.path.join(self.preprocessed_path, integrante)

            if not os.path.exists(integrante_path):
                continue

            os.makedirs(output_integrante_path, exist_ok=True)
            imagenes = [f for f in os.listdir(integrante_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            print(f"\n🔍 Procesando: {self._acortar_nombre(integrante)}")
            print(f"   Imágenes a procesar: {len(imagenes)}")

            for i, img_file in enumerate(imagenes):
                img_path = os.path.join(integrante_path, img_file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                # Convertir a escala de grises para detección
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detectar rostros
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                if len(faces) > 0:
                    # Tomar el rostro más grande
                    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                    x, y, w, h = faces[0]

                    # Expandir ligeramente el área del rostro
                    expand_x = int(w * 0.1)
                    expand_y = int(h * 0.1)
                    x = max(0, x - expand_x)
                    y = max(0, y - expand_y)
                    w = min(img.shape[1] - x, w + 2 * expand_x)
                    h = min(img.shape[0] - y, h + 2 * expand_y)

                    # Recortar y redimensionar
                    face = img[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, self.output_size)

                    # Guardar rostro recortado
                    output_file = f"rostro_{i:03d}.jpg"
                    output_path = os.path.join(output_integrante_path, output_file)
                    cv2.imwrite(output_path, face_resized)

                    rostros_detectados += 1
                else:
                    rostros_no_detectados += 1

                # Mostrar progreso
                if (i + 1) % 20 == 0:
                    print(f"   Procesadas: {i + 1}/{len(imagenes)}")

        self.estadisticas_preprocesamiento['rostros_detectados'] = rostros_detectados
        self.estadisticas_preprocesamiento['rostros_no_detectados'] = rostros_no_detectados
        self.estadisticas_preprocesamiento['tasa_deteccion'] = rostros_detectados / (rostros_detectados + rostros_no_detectados) if (rostros_detectados + rostros_no_detectados) > 0 else 0

        print(f"\n📊 RESUMEN DETECCIÓN FACIAL:")
        print(f"   ✅ Rostros detectados: {rostros_detectados}")
        print(f"   ❌ Rostros no detectados: {rostros_no_detectados}")
        print(f"   📈 Tasa de detección: {self.estadisticas_preprocesamiento['tasa_deteccion']:.2%}")

        return rostros_detectados > 0

    def _acortar_nombre(self, nombre_completo):
        """Acorta el nombre para mejor visualización"""
        partes = nombre_completo.split('_')
        if len(partes) >= 2:
            return f"{partes[0]} {partes[1]}"
        return nombre_completo

    def aumentar_datos(self, factor_aumento=3):
        """Aplicar aumento de datos para balancear y robustecer"""
        print("\n=== AUMENTO DE DATOS ===")

        if not os.path.exists(self.preprocessed_path):
            print("❌ Primero debe ejecutar la detección facial")
            return False

        # Configurar generador de aumento
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

        # Crear directorio de aumento
        if os.path.exists(self.augmented_path):
            shutil.rmtree(self.augmented_path)
        os.makedirs(self.augmented_path)

        imagenes_originales_totales = 0
        imagenes_aumentadas_totales = 0

        for integrante in self.integrantes_oficiales:
            integrante_path = os.path.join(self.preprocessed_path, integrante)
            output_integrante_path = os.path.join(self.augmented_path, integrante)

            if not os.path.exists(integrante_path):
                continue

            os.makedirs(output_integrante_path, exist_ok=True)

            # Copiar imágenes originales
            imagenes = [f for f in os.listdir(integrante_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in imagenes:
                src_path = os.path.join(integrante_path, img_file)
                dst_path = os.path.join(output_integrante_path, f"original_{img_file}")
                shutil.copy2(src_path, dst_path)

            imagenes_originales = len(imagenes)
            imagenes_originales_totales += imagenes_originales

            # Generar imágenes aumentadas
            target_imagenes = imagenes_originales * factor_aumento
            imagenes_generadas = 0

            for img_file in imagenes:
                if imagenes_generadas >= target_imagenes:
                    break

                img_path = os.path.join(integrante_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
                img = img.reshape((1,) + img.shape)  # Añadir dimensión batch

                # Generar imágenes aumentadas
                for i, batch in enumerate(datagen.flow(img, batch_size=1)):
                    if imagenes_generadas >= target_imagenes:
                        break

                    augmented_img = batch[0].astype('uint8')
                    augmented_img_bgr = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)

                    output_file = f"augmented_{img_file.split('.')[0]}_{i:02d}.jpg"
                    output_path = os.path.join(output_integrante_path, output_file)
                    cv2.imwrite(output_path, augmented_img_bgr)

                    imagenes_generadas += 1

            imagenes_aumentadas_totales += imagenes_generadas
            print(f"   {self._acortar_nombre(integrante)}: {imagenes_originales} → {imagenes_originales + imagenes_generadas} imágenes")

        self.estadisticas_preprocesamiento['imagenes_originales'] = imagenes_originales_totales
        self.estadisticas_preprocesamiento['imagenes_aumentadas'] = imagenes_aumentadas_totales
        self.estadisticas_preprocesamiento['total_con_aumento'] = imagenes_originales_totales + imagenes_aumentadas_totales

        print(f"\n📊 RESUMEN AUMENTO DE DATOS:")
        print(f"   📷 Imágenes originales: {imagenes_originales_totales}")
        print(f"   🎨 Imágenes aumentadas: {imagenes_aumentadas_totales}")
        print(f"   📈 Total después de aumento: {imagenes_originales_totales + imagenes_aumentadas_totales}")
        print(f"   🔁 Factor de aumento: {factor_aumento}x")

        return True

    def dividir_conjuntos(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Dividir en conjuntos de entrenamiento, validación y prueba"""
        print("\n=== DIVISIÓN DE CONJUNTOS ===")

        dataset_source = self.augmented_path if os.path.exists(self.augmented_path) else self.preprocessed_path
        datasets_path = os.path.join(os.path.dirname(self.dataset_path), 'datasets_finales')

        # Crear estructura de directorios
        train_path = os.path.join(datasets_path, 'train')
        val_path = os.path.join(datasets_path, 'validation')
        test_path = os.path.join(datasets_path, 'test')

        # Limpiar y crear directorios
        if os.path.exists(datasets_path):
            shutil.rmtree(datasets_path)

        for split_path in [train_path, val_path, test_path]:
            for integrante in self.integrantes_oficiales:
                os.makedirs(os.path.join(split_path, integrante), exist_ok=True)

        estadisticas_division = {}

        for integrante in self.integrantes_oficiales:
            integrante_path = os.path.join(dataset_source, integrante)

            if not os.path.exists(integrante_path):
                continue

            imagenes = [f for f in os.listdir(integrante_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not imagenes:
                continue

            # Dividir imágenes
            train_files, temp_files = train_test_split(
                imagenes,
                test_size=(val_ratio + test_ratio),
                random_state=42,
                stratify=None
            )

            val_files, test_files = train_test_split(
                temp_files,
                test_size=test_ratio/(val_ratio + test_ratio),
                random_state=42
            )

            # Copiar archivos a sus directorios
            for file_list, dest_path in [
                (train_files, train_path),
                (val_files, val_path),
                (test_files, test_path)
            ]:
                for f in file_list:
                    src = os.path.join(integrante_path, f)
                    dst = os.path.join(dest_path, integrante, f)
                    shutil.copy2(src, dst)

            estadisticas_division[integrante] = {
                'train': len(train_files),
                'validation': len(val_files),
                'test': len(test_files),
                'total': len(imagenes)
            }

        # Generar reporte de división
        print("\n📊 DISTRIBUCIÓN FINAL DE DATOS:")
        print("Integrante                Train   Val   Test   Total")
        print("-" * 50)

        total_train = 0
        total_val = 0
        total_test = 0
        total_total = 0

        for integrante, stats in estadisticas_division.items():
            nombre_corto = self._acortar_nombre(integrante)
            print(f"{nombre_corto:<20} {stats['train']:>6} {stats['validation']:>5} {stats['test']:>5} {stats['total']:>6}")

            total_train += stats['train']
            total_val += stats['validation']
            total_test += stats['test']
            total_total += stats['total']

        print("-" * 50)
        print(f"{'TOTAL':<20} {total_train:>6} {total_val:>5} {total_test:>5} {total_total:>6}")

        # Calcular porcentajes
        print(f"\n📈 PORCENTAJES:")
        print(f"   Entrenamiento: {total_train/total_total:.1%} ({total_train} imágenes)")
        print(f"   Validación:    {total_val/total_total:.1%} ({total_val} imágenes)")
        print(f"   Prueba:        {total_test/total_total:.1%} ({total_test} imágenes)")

        self.estadisticas_preprocesamiento['division'] = estadisticas_division
        self.estadisticas_preprocesamiento['totales'] = {
            'train': total_train,
            'validation': total_val,
            'test': total_test,
            'total': total_total
        }

        return datasets_path

    def visualizar_preprocesamiento(self):
        """Visualizar resultados del preprocesamiento"""
        if not self.estadisticas_preprocesamiento:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Resultados del Preprocesamiento - Fase 3', fontsize=16, fontweight='bold')

        # Gráfico 1: Detección facial
        if 'rostros_detectados' in self.estadisticas_preprocesamiento:
            detectados = self.estadisticas_preprocesamiento['rostros_detectados']
            no_detectados = self.estadisticas_preprocesamiento['rostros_no_detectados']

            labels = ['Detectados', 'No Detectados']
            sizes = [detectados, no_detectados]
            colors = ['lightgreen', 'lightcoral']

            axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Detección Facial', fontweight='bold')

        # Gráfico 2: Aumento de datos
        if 'imagenes_originales' in self.estadisticas_preprocesamiento:
            originales = self.estadisticas_preprocesamiento['imagenes_originales']
            aumentadas = self.estadisticas_preprocesamiento['imagenes_aumentadas']

            labels = ['Originales', 'Aumentadas']
            sizes = [originales, aumentadas]
            colors = ['lightblue', 'lightsteelblue']

            axes[0, 1].bar(labels, sizes, color=colors)
            axes[0, 1].set_title('Aumento de Datos', fontweight='bold')
            axes[0, 1].set_ylabel('Número de Imágenes')

            # Añadir valores en las barras
            for i, v in enumerate(sizes):
                axes[0, 1].text(i, v + 0.5, str(v), ha='center', va='bottom')

        # Gráfico 3: División de conjuntos
        if 'totales' in self.estadisticas_preprocesamiento:
            totales = self.estadisticas_preprocesamiento['totales']
            labels = ['Entrenamiento', 'Validación', 'Prueba']
            sizes = [totales['train'], totales['validation'], totales['test']]
            colors = ['gold', 'orange', 'coral']

            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('División de Conjuntos', fontweight='bold')

        # Información resumen
        axes[1, 1].axis('off')
        info_text = "RESUMEN PREPROCESAMIENTO\n\n"

        if 'tasa_deteccion' in self.estadisticas_preprocesamiento:
            info_text += f"Tasa detección: {self.estadisticas_preprocesamiento['tasa_deteccion']:.1%}\n"

        if 'total_con_aumento' in self.estadisticas_preprocesamiento:
            info_text += f"Imágenes totales: {self.estadisticas_preprocesamiento['total_con_aumento']}\n"

        if 'totales' in self.estadisticas_preprocesamiento:
            t = self.estadisticas_preprocesamiento['totales']
            info_text += f"\nConjunto final:\n"
            info_text += f"Train: {t['train']} ({t['train']/t['total']:.1%})\n"
            info_text += f"Val:   {t['validation']} ({t['validation']/t['total']:.1%})\n"
            info_text += f"Test:  {t['test']} ({t['test']/t['total']:.1%})\n"

        axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.show()

    def ejecutar_fase_completa(self):
        """Ejecutar toda la Fase 3 de manera secuencial"""
        print("🎯 INICIANDO FASE 3: PREPARACIÓN DE DATOS")
        print("=" * 60)

        # Verificar Fase 2
        if not self.verificar_fase_2():
            print("❌ No se puede continuar. Ejecuta la Fase 2 primero.")
            return None

        # 1. Detección y recorte facial
        if not self.detectar_y_recortar_rostros():
            print("❌ Error en detección facial")
            return None

        # 2. Aumento de datos
        if not self.aumentar_datos(factor_aumento=3):
            print("❌ Error en aumento de datos")
            return None

        # 3. División de conjuntos
        datasets_path = self.dividir_conjuntos()

        # 4. Visualizar resultados
        self.visualizar_preprocesamiento()

        print("\n✅ FASE 3 COMPLETADA EXITOSAMENTE!")
        print(f"📁 Datasets finales en: {datasets_path}")

        return datasets_path

# FUNCIÓN PRINCIPAL
def ejecutar_fase_3():
    """Función principal para ejecutar la Fase 3"""

    # Configurar rutas (igual que Fase 2)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = '/content/drive/MyDrive/dataset_facial'
        print("✅ Google Drive montado correctamente")
    except:
        print("⚠️  Ejecutando en entorno local")
        base_path = './dataset_facial'

    # Ejecutar Fase 3 completa
    preparador = DataPreparer(base_path, output_size=(224, 224))
    datasets_path = preparador.ejecutar_fase_completa()

    if datasets_path:
        print(f"\n🎯 PRÓXIMOS PASOS:")
        print("1. Revisar las estadísticas de preprocesamiento")
        print("2. Verificar la calidad de los rostros detectados")
        print("3. Proceder con la Fase 4: Modelado")
        print(f"4. Los datasets están en: {datasets_path}")

    return datasets_path

# EJECUTAR FASE 3
if __name__ == "__main__":
    ejecutar_fase_3()
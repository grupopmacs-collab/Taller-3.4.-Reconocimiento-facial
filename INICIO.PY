import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

class FacialRecognitionEDA:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.estadisticas = {}
        self.imagenes_ejemplo = {}

        # Lista oficial de integrantes del equipo
        self.integrantes_oficiales = [
            "Lizarraga_Aliaga_Alexis_Joaquin",
            "Huaychi_Sencia_Jose_Natividad",
            "Challco_Ccohuanqui_Samir_Junior",
            "Tejada_Taco_Francys_Raul",
            "Zavala_Huacarpuma_Juan_Aldair"
        ]

    def verificar_integrantes_con_imagenes(self):
        """Verifica qué integrantes tienen imágenes disponibles"""
        print("=== VERIFICACIÓN DE INTEGRANTES CON IMÁGENES ===")

        integrantes_con_imagenes = []
        integrantes_sin_imagenes = []

        for integrante in self.integrantes_oficiales:
            integrante_path = os.path.join(self.dataset_path, integrante)
            if os.path.exists(integrante_path) and os.path.isdir(integrante_path):
                imagenes = [f for f in os.listdir(integrante_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imagenes:
                    integrantes_con_imagenes.append(integrante)
                else:
                    integrantes_sin_imagenes.append(integrante)
            else:
                integrantes_sin_imagenes.append(integrante)

        self.estadisticas['integrantes_con_imagenes'] = integrantes_con_imagenes
        self.estadisticas['integrantes_sin_imagenes'] = integrantes_sin_imagenes
        self.estadisticas['total_con_imagenes'] = len(integrantes_con_imagenes)
        self.estadisticas['total_sin_imagenes'] = len(integrantes_sin_imagenes)

        print("✅ INTEGRANTES CON IMÁGENES:")
        for integrante in integrantes_con_imagenes:
            print(f"   - {integrante}")

        if integrantes_sin_imagenes:
            print("\n❌ INTEGRANTES SIN IMÁGENES (FALTANTES):")
            for integrante in integrantes_sin_imagenes:
                print(f"   - {integrante}")

        return integrantes_con_imagenes, integrantes_sin_imagenes

    def verificar_estructura_dataset(self):
        """Muestra la estructura esperada con nombres reales"""
        print("\n=== ESTRUCTURA REQUERIDA EN GOOGLE DRIVE ===")
        print("Tu Google Drive debe tener esta estructura exacta:\n")

        print("Google Drive/")
        print("└── MyDrive/")
        print("    └── dataset_facial/")
        for i, integrante in enumerate(self.integrantes_oficiales[:3], 1):
            print(f"        ├── {integrante}/")
            print(f"        │   ├── foto1.jpg")
            print(f"        │   ├── foto2.jpg")
            print(f"        │   └── ... (mínimo 100 imágenes)")
        print("        ├── ... (otros integrantes)")
        print("        └── [integrantes faltantes se crearán automáticamente]")

        input("\nPresiona Enter para continuar con el análisis...")

    def crear_estructura_faltante(self):
        """Crea carpetas para integrantes que no existen"""
        print("\n=== CREANDO ESTRUCTURA PARA INTEGRANTES FALTANTES ===")

        creados = []
        for integrante in self.estadisticas['integrantes_sin_imagenes']:
            integrante_path = os.path.join(self.dataset_path, integrante)
            if not os.path.exists(integrante_path):
                os.makedirs(integrante_path, exist_ok=True)
                creados.append(integrante)
                print(f"✅ Carpeta creada: {integrante}")

        if creados:
            print(f"\n📁 Se crearon {len(creados)} carpetas para integrantes faltantes")
            print("💡 Ahora puedes subir las imágenes a estas carpetas")
        else:
            print("✅ Todas las carpetas de integrantes ya existen")

        return creados

    def analizar_estructura_dataset(self):
        """Analiza la estructura y organización del dataset"""
        print("\n=== ANÁLISIS DETALLADO POR INTEGRANTE ===")

        estructura = {}
        total_imagenes = 0

        for integrante in self.estadisticas['integrantes_con_imagenes']:
            integrante_path = os.path.join(self.dataset_path, integrante)
            imagenes = [f for f in os.listdir(integrante_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            num_imagenes = len(imagenes)
            estructura[integrante] = num_imagenes
            total_imagenes += num_imagenes

            # Guardar imagen de ejemplo
            if imagenes:
                self.imagenes_ejemplo[integrante] = os.path.join(integrante_path, imagenes[0])

        self.estadisticas['estructura'] = estructura
        self.estadisticas['total_imagenes'] = total_imagenes
        self.estadisticas['balance'] = self._calcular_balance(estructura)

        print("📊 RESUMEN POR INTEGRANTE:")
        for integrante, count in estructura.items():
            # Formatear nombre para mejor visualización
            nombre_corto = self._acortar_nombre(integrante)
            estado = "✅ SUFICIENTE" if count >= 100 else "⚠️  INSUFICIENTE"
            progreso = "█" * min(count // 20, 5)  # Barra de progreso
            print(f"   {nombre_corto:<25} {count:>3} imágenes {progreso:<5} {estado}")

        print(f"\n📈 Total de imágenes en dataset: {total_imagenes}")
        print(f"🎯 Objetivo mínimo: {len(self.estadisticas['integrantes_con_imagenes'])} integrantes × 100 = {len(self.estadisticas['integrantes_con_imagenes']) * 100} imágenes")

        return estructura

    def _acortar_nombre(self, nombre_completo):
        """Acorta el nombre para mejor visualización"""
        partes = nombre_completo.split('_')
        if len(partes) >= 2:
            return f"{partes[0]} {partes[1]}"
        return nombre_completo

    def _calcular_balance(self, estructura):
        """Calcula qué tan balanceado está el dataset"""
        counts = list(estructura.values())
        if not counts:
            return 0
        balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0
        return balance_ratio

    def analizar_caracteristicas_imagenes(self):
        """Analiza dimensiones y características de las imágenes"""
        print("\n=== ANÁLISIS DE CALIDAD DE IMÁGENES ===")

        dimensiones = []
        canales = []

        for integrante in self.estadisticas['integrantes_con_imagenes'][:3]:  # Muestra de 3 integrantes
            integrante_path = os.path.join(self.dataset_path, integrante)
            imagenes = [f for f in os.listdir(integrante_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in imagenes[:10]:  # 10 imágenes por integrante
                img_path = os.path.join(integrante_path, img_file)
                img = cv2.imread(img_path)

                if img is not None:
                    h, w, c = img.shape
                    dimensiones.append((w, h))
                    canales.append(c)

        if dimensiones:
            dimensiones = np.array(dimensiones)
            self.estadisticas['dimensiones_promedio'] = dimensiones.mean(axis=0)
            self.estadisticas['dimensiones_min'] = dimensiones.min(axis=0)
            self.estadisticas['dimensiones_max'] = dimensiones.max(axis=0)

            print(f"✓ Dimensiones promedio: {self.estadisticas['dimensiones_promedio'][0]:.0f} × {self.estadisticas['dimensiones_promedio'][1]:.0f} px")
            print(f"✓ Rango de dimensiones: {self.estadisticas['dimensiones_min'][0]:.0f}×{self.estadisticas['dimensiones_min'][1]:.0f} a {self.estadisticas['dimensiones_max'][0]:.0f}×{self.estadisticas['dimensiones_max'][1]:.0f}")

            # Evaluar calidad basada en dimensiones
            dim_promedio = self.estadisticas['dimensiones_promedio'][0]
            if dim_promedio >= 500:
                calidad = "✅ ALTA"
            elif dim_promedio >= 300:
                calidad = "✅ ADECUADA"
            else:
                calidad = "⚠️  BAJA"
            print(f"✓ Calidad de resolución: {calidad}")

        return dimensiones

    def generar_reporte_completo(self):
        """Genera un reporte completo del EDA"""
        print("\n" + "="*70)
        print("           REPORTE COMPLETO - RECONOCIMIENTO FACIAL")
        print("="*70)

        # Configuración inicial
        self.verificar_estructura_dataset()
        integrantes_con, integrantes_sin = self.verificar_integrantes_con_imagenes()

        # Crear estructura faltante si es necesario
        if integrantes_sin:
            self.crear_estructura_faltante()

        if not integrantes_con:
            print("\n❌ No se encontraron imágenes para ningún integrante")
            print("💡 Por favor, sube imágenes al menos para 3 integrantes")
            return self.estadisticas

        # Ejecutar análisis
        estructura = self.analizar_estructura_dataset()
        self.analizar_caracteristicas_imagenes()

        # Evaluación final
        print("\n" + "="*50)
        print("EVALUACIÓN FINAL DEL PROYECTO")
        print("="*50)

        total_integrantes = len(integrantes_con)
        imagenes_por_integrante = list(estructura.values())
        integrantes_completos = sum(1 for img in imagenes_por_integrante if img >= 100)

        print(f"✅ Integrantes con imágenes: {total_integrantes}/5")
        print(f"✅ Integrantes con 100+ imágenes: {integrantes_completos}/{total_integrantes}")
        print(f"✅ Total de imágenes: {self.estadisticas['total_imagenes']}")
        print(f"✅ Balance del dataset: {self.estadisticas['balance']:.2f}")

        # Determinar estado del proyecto
        if total_integrantes >= 3 and integrantes_completos >= 3:
            estado = "✅ LISTO PARA ENTRENAMIENTO"
        elif total_integrantes >= 3:
            estado = "🟡 EN PROGRESO (necesita más imágenes)"
        else:
            estado = "🔴 INCOMPLETO (necesita más integrantes)"

        print(f"\n🎯 ESTADO DEL PROYECTO: {estado}")

        # Recomendaciones específicas
        print("\n📋 RECOMENDACIONES:")
        for integrante, count in estructura.items():
            if count < 100:
                nombre_corto = self._acortar_nombre(integrante)
                print(f"   📸 {nombre_corto}: necesita {100 - count} imágenes más")

        if integrantes_sin:
            print(f"\n   👥 Integrantes faltantes: {len(integrantes_sin)}")
            for integrante in integrantes_sin:
                nombre_corto = self._acortar_nombre(integrante)
                print(f"      - {nombre_corto}")

        return self.estadisticas

    def visualizar_dataset(self):
        """Genera visualizaciones del dataset"""
        if not self.estadisticas.get('integrantes_con_imagenes'):
            print("No hay datos para visualizar")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis del Dataset - Reconocimiento Facial', fontsize=16, fontweight='bold')

        # Gráfico 1: Distribución de imágenes por integrante
        if self.estadisticas.get('estructura'):
            integrantes = [self._acortar_nombre(name) for name in self.estadisticas['estructura'].keys()]
            counts = list(self.estadisticas['estructura'].values())

            colors = ['green' if count >= 100 else 'orange' for count in counts]
            bars = axes[0, 0].bar(integrantes, counts, color=colors)
            axes[0, 0].set_title('Imágenes por Integrante', fontweight='bold')
            axes[0, 0].set_ylabel('Número de Imágenes')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Añadir valores en las barras
            for bar, count in zip(bars, counts):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{count}', ha='center', va='bottom', fontsize=9)

        # Gráfico 2: Estado del proyecto
        integrantes_con = self.estadisticas.get('total_con_imagenes', 0)
        integrantes_sin = self.estadisticas.get('total_sin_imagenes', 0)

        labels = ['Con Imágenes', 'Sin Imágenes']
        sizes = [integrantes_con, integrantes_sin]
        colors = ['lightgreen', 'lightcoral']

        axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
        axes[0, 1].set_title('Cobertura de Integrantes', fontweight='bold')

        # Gráfico 3: Ejemplos de imágenes
        ejemplos = list(self.imagenes_ejemplo.items())[:2]
        for idx, (integrante, img_path) in enumerate(ejemplos):
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (200, 200))

                axes[1, idx].imshow(img_resized)
                nombre_corto = self._acortar_nombre(integrante)
                axes[1, idx].set_title(f'Ejemplo: {nombre_corto}', fontweight='bold')
                axes[1, idx].axis('off')

        # Información resumen
        axes[1, 1].axis('off')
        info_text = f"""
RESUMEN DEL PROYECTO

Integrantes totales: 5
Con imágenes: {integrantes_con}
Sin imágenes: {integrantes_sin}

Total imágenes: {self.estadisticas.get('total_imagenes', 0)}
Imágenes necesarias: {integrantes_con * 100}

Estado: {'✅ LISTO' if integrantes_con >= 3 and self.estadisticas.get('total_imagenes', 0) >= 300 else '🔄 EN PROGRESO'}
        """
        axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.show()

# FUNCIÓN PRINCIPAL
def ejecutar_analisis_completo():
    """Función principal que ejecuta todo el análisis"""

    print("🔍 ANÁLISIS EXPLORATORIO - RECONOCIMIENTO FACIAL")
    print("="*60)
    print("EQUIPO:")
    print("  • Lizarraga Aliaga Alexis Joaquin")
    print("  • Huaychi Sencia José Natividad")
    print("  • Challco Ccohuanqui Samir Junior")
    print("  • Tejada Taco Francys Raul")
    print("  • Zavala Huacarpuma Juan Aldair")
    print("="*60)

    # Configurar rutas
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = '/content/drive/MyDrive/dataset_facial'
        print("✅ Google Drive montado correctamente")
    except:
        print("⚠️  Ejecutando en entorno local")
        base_path = './dataset_facial'

    # Verificar si existe el dataset
    if not os.path.exists(base_path):
        print(f"\n📁 Creando estructura base en: {base_path}")
        os.makedirs(base_path, exist_ok=True)

    # Ejecutar análisis
    eda = FacialRecognitionEDA(base_path)
    resultados = eda.generar_reporte_completo()

    if resultados.get('integrantes_con_imagenes'):
        eda.visualizar_dataset()

    print("\n🎯 PRÓXIMOS PASOS:")
    integrantes_con = resultados.get('total_con_imagenes', 0)
    if integrantes_con >= 3:
        print("✅ Puedes proceder con el preprocesamiento de imágenes")
        print("💡 Ejecuta este código nuevamente cuando agregues más imágenes")
    else:
        print("📝 Necesitas al menos 3 integrantes con imágenes para comenzar")

    print("\n🔄 Para actualizar: Agrega imágenes y ejecuta este mismo código")

# EJECUTAR EL ANÁLISIS
if __name__ == "__main__":
    ejecutar_analisis_completo()
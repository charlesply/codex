import sys
import os
import json
import re
from difflib import SequenceMatcher
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QProgressBar, QGroupBox, QRadioButton, QCheckBox, QSplitter,
    QDialog, QDialogButtonBox, QSpinBox, QTabWidget, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor

import openai
from openai import OpenAI


@dataclass
class CountryLanguage:
    """Classe para armazenar informações de país e idioma"""
    code: str
    country: str
    language: str


class ConfigDialog(QDialog):
    """Diálogo para configurar associações de sigla, país e idioma"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurar Siglas")
        self.setMinimumSize(600, 400)
        self.associations = self.load_associations()
        self.init_ui()
        self.apply_theme()
        
    def apply_theme(self):
        """Aplica tema claro ao diálogo"""
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                color: #333333;
            }
            QTableWidget {
                background-color: #ffffff;
                color: #333333;
                gridline-color: #dddddd;
                border: 1px solid #cccccc;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                color: #333333;
                padding: 5px;
                border: 1px solid #cccccc;
            }
            QPushButton {
                background-color: #10a37f;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0d8f6f;
            }
        """)
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Tabela para exibir e editar associações
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Sigla", "País", "Idioma"])
        self.table.horizontalHeader().setStretchLastSection(True)
        
        # Carregar associações existentes
        self.load_table()
        
        layout.addWidget(QLabel("Configurar associações (uma por linha):"))
        layout.addWidget(self.table)
        
        # Botões para adicionar/remover linhas
        btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("➕ Adicionar Linha")
        self.btn_add.clicked.connect(self.add_row)
        
        self.btn_remove = QPushButton("➖ Remover Linha")
        self.btn_remove.clicked.connect(self.remove_row)
        
        self.btn_reset = QPushButton("🔄 Restaurar Padrão")
        self.btn_reset.clicked.connect(self.reset_defaults)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        # Botões OK/Cancelar
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save_and_accept)
        buttons.rejected.connect(self.reject)
        
        layout.addWidget(buttons)
        self.setLayout(layout)
        
    def load_associations(self) -> Dict[str, CountryLanguage]:
        """Carrega associações salvas ou retorna padrões"""
        settings = QSettings('StoryGenerator', 'Associations')
        saved = settings.value('associations', {})
        
        if not saved:
            # Associações padrão baseadas no exemplo
            defaults = {
                'CRO': CountryLanguage('CRO', 'Croácia', 'Croata'),
                'GRE': CountryLanguage('GRE', 'Grécia', 'Grego'),
                'ARA': CountryLanguage('ARA', 'Arábia Saudita', 'Árabe'),
                'CZE': CountryLanguage('CZE', 'República Tcheca', 'Tcheco'),
                'BRA': CountryLanguage('BRA', 'Brasil', 'Português'),
                'POR': CountryLanguage('POR', 'Portugal', 'Português'),
                'USA': CountryLanguage('USA', 'Estados Unidos', 'Inglês'),
                'MEX': CountryLanguage('MEX', 'México', 'Espanhol'),
                'ESP': CountryLanguage('ESP', 'Espanha', 'Espanhol'),
                'FRA': CountryLanguage('FRA', 'França', 'Francês'),
                'GER': CountryLanguage('GER', 'Alemanha', 'Alemão'),
                'ITA': CountryLanguage('ITA', 'Itália', 'Italiano'),
                'ING': CountryLanguage('ING', 'Reino Unido', 'Inglês'),
                'ROM': CountryLanguage('ROM', 'Romênia', 'Romeno'),
            }
            return defaults
        return saved
        
    def load_table(self):
        """Carrega associações na tabela"""
        self.table.setRowCount(len(self.associations))
        for i, (code, cl) in enumerate(self.associations.items()):
            self.table.setItem(i, 0, QTableWidgetItem(code))
            self.table.setItem(i, 1, QTableWidgetItem(cl.country))
            self.table.setItem(i, 2, QTableWidgetItem(cl.language))
            
    def add_row(self):
        """Adiciona nova linha à tabela"""
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        
    def remove_row(self):
        """Remove linha selecionada"""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)
            
    def reset_defaults(self):
        """Restaura associações padrão"""
        reply = QMessageBox.question(
            self, 'Confirmar', 
            'Restaurar configurações padrão?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.associations = {}
            self.associations = self.load_associations()
            self.load_table()
            
    def save_and_accept(self):
        """Salva associações e fecha diálogo"""
        new_associations = {}
        for row in range(self.table.rowCount()):
            code_item = self.table.item(row, 0)
            country_item = self.table.item(row, 1)
            language_item = self.table.item(row, 2)
            
            if code_item and country_item and language_item:
                code = code_item.text().strip().upper()
                if code:
                    new_associations[code] = CountryLanguage(
                        code,
                        country_item.text().strip(),
                        language_item.text().strip()
                    )
        
        # Salvar nas configurações
        settings = QSettings('StoryGenerator', 'Associations')
        settings.setValue('associations', new_associations)
        self.accept()


class StoryQualityAnalyzer:
    """Analisador de qualidade das histórias geradas"""

    def __init__(self, language: str, country: str):
        self.language = language
        self.country = country

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcula a similaridade aproximada entre dois textos."""
        tokens1 = self._tokenize_for_similarity(text1)
        tokens2 = self._tokenize_for_similarity(text2)

        if not tokens1 or not tokens2:
            return 0.0

        matcher = SequenceMatcher(None, tokens1, tokens2)
        return matcher.ratio()

    @staticmethod
    def _tokenize_for_similarity(text: str) -> List[str]:
        """Normaliza o texto para comparação removendo pontuação e caixa."""
        return re.findall(r"\w+", text.lower())

    def analyze_story(self, story_text: str, chapters: list, title: str) -> dict:
        """Analisa a qualidade da história gerada"""
        analysis = {
            'score': 0,
            'issues': [],
            'warnings': [],
            'strengths': [],
            'language_check': True,
            'repetition_check': True,
            'plot_progression': True,
            # 'title_adherence': True, # REMOVIDO
        }
        
        # 1. Verificar idioma
        language_issues = self.check_language_consistency(story_text)
        if language_issues:
            analysis['language_check'] = False
            analysis['issues'].extend(language_issues)
            analysis['score'] -= 3
        
        # 2. Verificar repetições entre capítulos
        repetitions = self.check_chapter_repetitions(chapters)
        if repetitions:
            analysis['repetition_check'] = False
            analysis['issues'].extend(repetitions)
            analysis['score'] -= 2
        
        # 3. Verificar progressão da trama
        progression_issues = self.check_plot_progression(chapters)
        if progression_issues:
            analysis['plot_progression'] = False
            analysis['warnings'].extend(progression_issues)
            analysis['score'] -= 1
        
        # 4. Verificar aderência ao título - REMOVIDO
        # title_issues = self.check_title_adherence(story_text, title)
        # if title_issues:
        #     analysis['title_adherence'] = False
        #     analysis['issues'].extend(title_issues)
        #     analysis['score'] -= 2
        
        # 5. Calcular pontos positivos
        analysis['strengths'] = self.identify_strengths(story_text, chapters)
        analysis['score'] += len(analysis['strengths'])
        
        # Normalizar score (0-10)
        analysis['score'] = max(0, min(10, analysis['score'] + 7))
        
        return analysis
    
    def check_language_consistency(self, text: str) -> list:
        """Verifica se o texto está no idioma correto"""
        issues = []
        
        # Marcadores de português que não devem aparecer
        portuguese_markers = [
            'CAPÍTULO', 'CAPITOL', 'FIM DO', 'FIN DO',
            'capítulo', 'capitol', 'fim do', 'fin do'
        ]
        
        # Verificar amostra do texto
        sample = text[:2000].lower()
        
        for marker in portuguese_markers:
            # CORREÇÃO: Verificar se o marcador está como uma palavra isolada para evitar falsos positivos
            if re.search(r'\b' + re.escape(marker.lower()) + r'\b', sample):
                issues.append(f"Texto contém marcador em português: '{marker}'")
        
        # Verificar densidade de palavras comuns em português
        portuguese_common = ['que', 'para', 'com', 'por', 'mas', 'muito', 'quando', 'depois']
        pt_count = sum(1 for word in portuguese_common if word in sample.split())
        
        if pt_count > 10:
            issues.append(f"Alta densidade de palavras em português detectada ({pt_count} palavras)")
        
        return issues
    
    def check_chapter_repetitions(self, chapters: list) -> list:
        """Verifica repetições significativas entre capítulos"""
        issues = []
        
        for i in range(len(chapters) - 1):
            for j in range(i + 1, len(chapters)):
                similarity = self.calculate_similarity(chapters[i], chapters[j])
                if similarity > 0.7:  # 70% similar
                    issues.append(f"Capítulos {i+1} e {j+1} são muito similares ({similarity:.0%})")
        
        return issues
    
    def check_plot_progression(self, chapters: list) -> list:
        """Verifica se a trama progride adequadamente"""
        warnings = []
        
        # Verificar tamanho dos capítulos
        for i, chapter in enumerate(chapters, 1):
            word_count = len(chapter.split())
            if word_count < 500:
                warnings.append(f"Capítulo {i} muito curto ({word_count} palavras)")
            elif word_count > 1500:
                warnings.append(f"Capítulo {i} muito longo ({word_count} palavras)")
        
        # Verificar se há desenvolvimento (mudança de contexto)
        static_chapters = []
        for i in range(1, len(chapters)):
            if self.calculate_similarity(chapters[i-1], chapters[i]) > 0.5:
                static_chapters.append(i+1)
        
        if static_chapters:
            warnings.append(f"Capítulos sem progressão clara: {static_chapters}")
        
        return warnings
    
    def check_title_adherence(self, text: str, title: str) -> list:
        """Verifica se a história segue o título dado"""
        issues = []
        
        # CORREÇÃO: Extrair palavras-chave dinamicamente do título, em vez de usar uma lista 'hardcoded'
        
        # Lista de stopwords comuns (português e inglês) para ignorar
        stopwords = set([
            'a', 'o', 'e', 'ou', 'em', 'de', 'do', 'da', 'dos', 'das', 'com', 'para', 'por', 
            'um', 'uma', 'que', 'se', 'mas', 'como', 'seu', 'sua', 'meu', 'minha',
            'at', 'my', 'in', 'on', 'his', 'her', 'and', 'the', 'to', 'as', 'i', 
            'he', 'she', 'it', 'was', 'is', 'what', 'then', 'up', 'our', 'me'
        ])
        
        # Limpar e extrair palavras-chave do título
        title_lower_cleaned = re.sub(r'[^\w\s]', '', title.lower())  # Remove pontuação
        all_words = title_lower_cleaned.split()
        key_elements = [word for word in all_words if word not in stopwords and len(word) > 3]

        # Verificar presença dos elementos na história
        text_lower = text.lower()
        missing_elements = []
        
        # CORREÇÃO DE INDENTAÇÃO: O loop 'for' deve vir ANTES da lógica 'found'
        for element in key_elements:
            variations = element.split('/')
            found = any(var in text_lower for var in variations)
            if not found:
                missing_elements.append(element)
        
        if missing_elements:
            issues.append(f"Elementos do título ausentes na história: {', '.join(missing_elements)}")
        
        return issues
    
    def identify_strengths(self, text: str, chapters: list) -> list:
        """Identifica pontos fortes da história"""
        strengths = []
        
        # Verificar gancho inicial
        if chapters and len(chapters[0]) > 300:
            first_chapter_words = chapters[0][:500].lower()
            if any(word in first_chapter_words for word in ['choque', 'surpresa', 'inesperado', 'shock', 'surprise']):
                strengths.append("Gancho inicial impactante")
        
        # Verificar presença de diálogos
        if '"' in text or "'" in text or '—' in text:
            strengths.append("Presença de diálogos")
        
        # Verificar variedade de emoções
        emotions = ['amor', 'ódio', 'medo', 'alegria', 'tristeza', 'raiva', 'love', 'hate', 'fear', 'joy', 'sad', 'anger']
        emotion_count = sum(1 for emotion in emotions if emotion in text.lower())
        if emotion_count >= 3:
            strengths.append("Rica em emoções variadas")
        
        # Verificar tamanho adequado
        total_words = len(text.split())
        if 7000 <= total_words <= 12000:
            strengths.append("Tamanho adequado para narração")
        
        return strengths


class APILogger:
    """Sistema de logging para chamadas da API"""
    
    def __init__(self, log_dir: str = "API_Logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{self.session_id}.json"
        self.logs = []
        
    def log_api_call(self, prompt: str, response: str, context: dict = None):
        """Registra uma chamada da API"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt[:500] + '...' if len(prompt) > 500 else prompt,
            'prompt_full_length': len(prompt),
            'response': response[:500] + '...' if len(response) > 500 else response,
            'response_full_length': len(response),
            'context': context or {},
            'tokens_estimate': (len(prompt) + len(response)) // 4  # Estimativa rough
        }
        
        self.logs.append(log_entry)
        self.save_logs()
        
    def save_logs(self):
        """Salva logs em arquivo JSON"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
    
    def get_session_summary(self) -> dict:
        """Retorna resumo da sessão"""
        return {
            'total_calls': len(self.logs),
            'total_tokens_estimate': sum(log['tokens_estimate'] for log in self.logs),
            'session_duration': self.logs[-1]['timestamp'] if self.logs else None,
            'log_file': str(self.log_file)
        }


class StoryGeneratorThread(QThread):
    """Thread para gerar histórias sem travar a interface"""
    progress = pyqtSignal(str)
    log_message = pyqtSignal(str)
    api_log = pyqtSignal(str, str)  # prompt, response
    story_generated = pyqtSignal(str, str, str)  # título, nome_arquivo, conteúdo
    checklist_update = pyqtSignal(str, dict)  # título, checklist_data
    quality_report = pyqtSignal(str, dict)  # título, análise
    finished_all = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, api_key: str, titles: List[str], style: str, 
                 country: str, language: str, base_name: str, start_num: int):
        super().__init__()
        self.api_key = api_key
        self.titles = titles
        self.style = style
        self.country = country
        self.language = language
        self.base_name = base_name
        self.start_num = start_num
        self.client = None
        self.api_logger = APILogger()
        self.quality_analyzer = StoryQualityAnalyzer(language, country)
        
    def run(self):
        """Executa geração de histórias"""
        try:
            self.client = OpenAI(api_key=self.api_key)
            
            # Criar pasta Roteiros Prontos se não existir
            output_dir = Path("Roteiros Prontos")
            output_dir.mkdir(exist_ok=True)
            
            for i, title in enumerate(self.titles):
                current_num = self.start_num + i
                file_base = f"{self.base_name}-{current_num:03d}"
                
                self.progress.emit(f"Processando história {i+1}/{len(self.titles)}: {title}")
                self.log_message.emit(f"🌍 Gerando história em {self.language} para {self.country}")
                self.log_message.emit(f"📝 Título: {title}")
                
                # Checklist para acompanhamento
                checklist = {
                    "title": title,
                    "options": "⏳ Aguardando",
                    "structure": "⏳ Aguardando",
                    "chapters": [],
                    "hook": "⏳ Aguardando",
                    "conclusion": "⏳ Aguardando",
                    "title_translated": "⏳ Aguardando",
                    # "description": "⏳ Aguardando", # REMOVIDO
                    "quality_check": "⏳ Aguardando",
                    "narrar_file": "⏳ Aguardando",
                    "srt_file": "⏳ Aguardando",
                    "docs_file": "⏳ Aguardando"
                }
                
                for j in range(1, 9):
                    checklist["chapters"].append({f"chapter_{j}": "⏳ Aguardando"})
                
                self.checklist_update.emit(title, checklist)
                
                # Gerar história completa com todos os componentes
                docs_content = self.generate_full_story(title, file_base, checklist)
                
                # Emitir sinal com a história gerada
                self.story_generated.emit(title, f"{file_base}-DOCS.txt", docs_content)
                
                # Pequena pausa entre gerações
                time.sleep(2)
                
            self.finished_all.emit()
            
        except Exception as e:
            self.error.emit(str(e))
            
    def generate_full_story(self, title: str, file_base: str, checklist: dict) -> str:
        """Gera uma história completa com todos os capítulos"""
        prompts_dir = Path(f"Prompts/{self.style}")
        
        # VERIFICAR SE A PASTA EXISTE
        if not prompts_dir.exists():
            self.error.emit(f"ERRO: Pasta de prompts não existe: {prompts_dir}")
            return ""
        
        # Sistema de contexto para manter continuidade
        conversation_history = []
        
        # 1. Gerar opções de trama - USANDO SEU ARQUIVO
        self.log_message.emit("📝 Gerando opções de trama...")
        self.log_message.emit(f"📁 Usando arquivo: {prompts_dir / 'tenho_um_canal.txt'}")
        checklist["options"] = "🔄 Processando"
        self.checklist_update.emit(title, checklist)
        
        prompt1 = self.load_prompt(prompts_dir / "tenho_um_canal.txt", title)
        
        # VERIFICAÇÃO CRÍTICA
        if not prompt1 or len(prompt1) < 100:
            self.error.emit("ERRO: Arquivo tenho_um_canal.txt não carregado corretamente!")
            return ""
        
        response1 = self.send_to_gpt(prompt1, context={'step': 'options', 'title': title})
        conversation_history.append({"role": "user", "content": prompt1})
        conversation_history.append({"role": "assistant", "content": response1})
        
        checklist["options"] = "✅ Concluído"
        self.checklist_update.emit(title, checklist)
        
        # 2. Selecionar opção
        self.log_message.emit("🎯 Selecionando melhor opção...")
        prompt2 = self.load_prompt(prompts_dir / "selecionar_opcao.txt", title)
        conversation_history.append({"role": "user", "content": prompt2})
        response2 = self.send_to_gpt_with_context(conversation_history, context={'step': 'selection'})
        conversation_history.append({"role": "assistant", "content": response2})
        
        # 3. Criar estrutura dos 8 capítulos
        self.log_message.emit(f"📋 Criando estrutura dos capítulos em {self.language}...")
        checklist["structure"] = "🔄 Processando"
        self.checklist_update.emit(title, checklist)
        
        prompt3 = self.load_prompt(prompts_dir / "criar_estrutura.txt", title)
        
        # REMOVIDO: Bloco hardcoded de regras da Etapa 3 foi removido
        # Suas regras agora estão no criar_estrutura.txt
        
        conversation_history.append({"role": "user", "content": prompt3})
        structure = self.send_to_gpt_with_context(conversation_history, context={'step': 'structure'})
        conversation_history.append({"role": "assistant", "content": structure})
        
        checklist["structure"] = "✅ Concluído"
        self.checklist_update.emit(title, checklist)
        
        # 4. Desenvolver cada capítulo com os prompts corretos
        chapters_text = []
        chapters_raw = []  # Para análise com marcadores
        
        # --- ETAPA 4a: GERAR CAPÍTULO 1 (usando texto_claude.txt) ---
        
        self.log_message.emit(f"📖 Desenvolvendo Capítulo 1 (Mestre) em {self.language}...")
        checklist["chapters"][0] = {"chapter_1": "🔄 Processando"}
        self.checklist_update.emit(title, checklist)
        
        # Carregar o prompt mestre (texto_claude.txt)
        prompt_chap1_path = prompts_dir / "texto_claude.txt"
        chapter_1_prompt = self.load_prompt(prompt_chap1_path, title)
        
        if not chapter_1_prompt:
            self.error.emit("ERRO CRÍTICO: Não foi possível carregar 'texto_claude.txt'")
            return ""
            
        # Injetar a estrutura de capítulos que foi gerada na Etapa 3
        # (O seu prompt 'texto_claude.txt' espera a variável [CAPITULOS])
        chapter_1_prompt = chapter_1_prompt.replace('[CAPITULOS]', structure)
        
        # Enviar para a API (adicionando ao histórico)
        conversation_history.append({"role": "user", "content": chapter_1_prompt})
        chapter_1_response = self.send_to_gpt_with_context(
            conversation_history, 
            context={'step': 'chapter_1_master', 'language': self.language}
        )
        conversation_history.append({"role": "assistant", "content": chapter_1_response})
        
        # Guardar versão raw e limpa
        chapters_raw.append(chapter_1_response)
        clean_chapter_1 = self.clean_chapter_for_narration(chapter_1_response)
        chapters_text.append(clean_chapter_1)
        
        checklist["chapters"][0] = {"chapter_1": "✅ Concluído"}
        self.checklist_update.emit(title, checklist)
        
        time.sleep(1)  # Pausa

        # --- ETAPA 4b: GERAR CAPÍTULOS 2-8 (usando desenvolve-caps-2a8.txt) ---
        
        for chapter_num in range(2, 9):
            self.log_message.emit(f"📖 Desenvolvendo Capítulo {chapter_num} em {self.language}...")
            checklist["chapters"][chapter_num-1] = {f"chapter_{chapter_num}": "🔄 Processando"}
            self.checklist_update.emit(title, checklist)
            
            # Carregar o prompt de continuação (desenvolve-caps-2a8.txt)
            prompt_loop_path = prompts_dir / "desenvolve-caps-2a8.txt"
            chapter_loop_prompt = self.load_prompt(prompt_loop_path, title)
            
            if not chapter_loop_prompt:
                self.error.emit(f"ERRO CRÍTICO: Não foi possível carregar 'desenvolve-caps-2a8.txt' para o Cap {chapter_num}")
                continue # Pula para o próximo capítulo
            
            # Substituir as variáveis dinâmicas do prompt
            # O seu arquivo usa 'CHAPTER_NUM', 'Capítulo X' e 'CAPÍTULO X'
            chapter_loop_prompt = chapter_loop_prompt.replace('CHAPTER_NUM', str(chapter_num))
            chapter_loop_prompt = chapter_loop_prompt.replace('Capítulo X', f'Capítulo {chapter_num}')
            chapter_loop_prompt = chapter_loop_prompt.replace('CAPÍTULO X', f'CAPÍTULO {chapter_num}')
            
            # Adicionar o trecho da ESTRUTURA para dar contexto (ESSENCIAL!)
            structure_snippet = self.extract_chapter_from_structure(structure, chapter_num)
            
            final_chapter_prompt = f"""{chapter_loop_prompt}

REGRAS ADICIONAIS DE CONTEXTO:
1. Continue a história mantendo 100% de consistência com os capítulos anteriores.
2. País/Contexto: {self.country}
3. Idioma: {self.language}
4. FOCO DESTE CAPÍTULO (Baseado na estrutura): {structure_snippet}

Escreva o capítulo {chapter_num} agora:"""
            
            # Enviar para a API (adicionando ao histórico)
            conversation_history.append({"role": "user", "content": final_chapter_prompt})
            chapter_response = self.send_to_gpt_with_context(
                conversation_history, 
                context={'step': f'chapter_{chapter_num}', 'language': self.language}
            )
            conversation_history.append({"role": "assistant", "content": chapter_response})
            
            # Guardar versão raw e limpa
            chapters_raw.append(chapter_response)
            clean_chapter = self.clean_chapter_for_narration(chapter_response)
            chapters_text.append(clean_chapter)
            
            checklist["chapters"][chapter_num-1] = {f"chapter_{chapter_num}": "✅ Concluído"}
            self.checklist_update.emit(title, checklist)
            
            time.sleep(1)  # Pequena pausa para não sobrecarregar API
        
        # 5. Gerar Hook (usando hook.txt)
        self.log_message.emit(f"🎣 Gerando hook em {self.language}...")
        checklist["hook"] = "🔄 Processando"
        self.checklist_update.emit(title, checklist)
        
        hook_prompt_path = prompts_dir / "hook.txt"
        hook_prompt = self.load_prompt(hook_prompt_path, title)
        
        if not hook_prompt:
            self.error.emit("ERRO CRÍTICO: Não foi possível carregar 'hook.txt'")
            clean_hook = "(Erro ao carregar hook.txt)"
        else:
            # Adicionar contexto do início da história para o hook
            contexto_historia = f"Contexto da história (Capítulo 1):\n{chapters_text[0][:1000]}"
            hook_prompt_com_contexto = f"{hook_prompt}\n\n{contexto_historia}"
            
            hook_response = self.send_to_gpt(hook_prompt_com_contexto, context={'step': 'hook'})
            # A função remove_all_markers vai limpar os marcadores INI/END
            clean_hook = self.remove_all_markers(hook_response)
        
        checklist["hook"] = "✅ Concluído"
        self.checklist_update.emit(title, checklist)
        
        # 6. Gerar Conclusão (usando conclusao.txt)
        self.log_message.emit(f"🎬 Gerando conclusão em {self.language}...")
        checklist["conclusion"] = "🔄 Processando"
        self.checklist_update.emit(title, checklist)
        
        conclusion_prompt_path = prompts_dir / "conclusao.txt"
        conclusion_prompt = self.load_prompt(conclusion_prompt_path, title)
        
        if not conclusion_prompt:
            self.error.emit("ERRO CRÍTICO: Não foi possível carregar 'conclusao.txt'")
            clean_conclusion = "(Erro ao carregar conclusao.txt)"
        else:
            # Adicionar contexto do final da história para a conclusão
            contexto_final = f"Contexto do final da história (Capítulo 8):\n{chapters_text[-1][-1000:]}"
            conclusion_prompt_com_contexto = f"{conclusion_prompt}\n\n{contexto_final}"
            
            conclusion_response = self.send_to_gpt(conclusion_prompt_com_contexto, context={'step': 'conclusion'})
            # A função remove_all_markers vai limpar os marcadores INI/END
            clean_conclusion = self.remove_all_markers(conclusion_response)
            
        checklist["conclusion"] = "✅ Concluído"
        self.checklist_update.emit(title, checklist)
        
        # 7. Traduzir/Adaptar título (usando titulo_traduzido.txt)
        self.log_message.emit(f"🌐 Adaptando título para {self.language}...")
        checklist["title_translated"] = "🔄 Processando"
        self.checklist_update.emit(title, checklist)
        
        if self.language.lower() != "português":
            title_prompt_path = prompts_dir / "titulo_traduzido.txt"
            title_prompt = self.load_prompt(title_prompt_path, title)
            
            if not title_prompt:
                self.error.emit("ERRO CRÍTICO: Não foi possível carregar 'titulo_traduzido.txt'")
                title_translated = title # Usa o original como fallback
            else:
                title_translated = self.send_to_gpt(title_prompt, context={'step': 'title_translation'}).strip()
        else:
            title_translated = title
        
        checklist["title_translated"] = "✅ Concluído"
        self.checklist_update.emit(title, checklist)
        
        # 8. Gerar descrição YouTube - ETAPA REMOVIDA
        
        # 9. ANÁLISE DE QUALIDADE
        self.log_message.emit("🔍 Analisando qualidade da história...")
        checklist["quality_check"] = "🔄 Processando"
        self.checklist_update.emit(title, checklist)
        
        # Montar texto completo para análise
        full_story = ""
        if clean_hook:
            full_story += clean_hook + "\n\n"
        for chapter in chapters_text:
            full_story += chapter + "\n\n"
        if clean_conclusion:
            full_story += clean_conclusion
        
        # Realizar análise
        quality_analysis = self.quality_analyzer.analyze_story(
            full_story,
            chapters_text,
            title
        )
        
        # Emitir relatório de qualidade
        self.quality_report.emit(title, quality_analysis)
        
        # Log da análise
        self.log_message.emit(f"📊 Nota de qualidade: {quality_analysis['score']}/10")
        if quality_analysis['issues']:
            self.log_message.emit(f"⚠️ Problemas encontrados: {len(quality_analysis['issues'])}")
            for issue in quality_analysis['issues'][:3]:  # Mostrar até 3 problemas
                self.log_message.emit(f"  - {issue}")
        
        checklist["quality_check"] = f"✅ Nota: {quality_analysis['score']}/10"
        self.checklist_update.emit(title, checklist)
        
        # 10. Montar texto final para narração (-NARRAR.txt)
        self.log_message.emit("📚 Montando texto final para narração...")
        
        narration_text = full_story  # Já está limpo e sem marcadores
        
        # Verificar tamanho
        word_count = len(narration_text.split())
        self.log_message.emit(f"📊 Total de palavras: {word_count}")
        
        # 11. Salvar arquivo -NARRAR.txt
        narrar_filename = f"{file_base}-NARRAR.txt"
        narrar_path = Path("Roteiros Prontos") / narrar_filename
        with open(narrar_path, 'w', encoding='utf-8') as f:
            f.write(narration_text)
        
        checklist["narrar_file"] = "✅ Salvo"
        self.checklist_update.emit(title, checklist)
        self.log_message.emit(f"💾 Arquivo salvo: {narrar_filename}")
        
        # 12. Gerar SRT
        self.log_message.emit("🎬 Gerando arquivo SRT...")
        srt_filename = f"{file_base}-NARRAR.srt"
        self.generate_srt(narration_text, srt_filename)
        
        checklist["srt_file"] = "✅ Salvo"
        self.checklist_update.emit(title, checklist)
        
        # 13. Montar conteúdo completo para -DOCS.txt
        full_chapters_with_structure = ""
        for i, chapter_raw in enumerate(chapters_raw, 1):
            full_chapters_with_structure += f"\n{chapter_raw}\n"
            full_chapters_with_structure += "-" * 50 + "\n"
        
        docs_content = f"""=== INFORMAÇÕES DO ROTEIRO ===
Título Original: {title}
Título Traduzido: {title_translated}
País: {self.country}
Idioma: {self.language}
Data de Geração: {datetime.now().strftime('%d/%m/%Y %H:%M')}

=== ANÁLISE DE QUALIDADE ===
Nota: {quality_analysis['score']}/10
Problemas: {', '.join(quality_analysis['issues']) if quality_analysis['issues'] else 'Nenhum'}
Avisos: {', '.join(quality_analysis['warnings']) if quality_analysis['warnings'] else 'Nenhum'}
Pontos Fortes: {', '.join(quality_analysis['strengths']) if quality_analysis['strengths'] else 'Nenhum'}

=== ESTRUTURA DOS CAPÍTULOS ===
{structure}

=== CAPÍTULOS DESENVOLVIDOS (COM MARCADORES) ===
{full_chapters_with_structure}

=== HOOK (LIMPO) ===
{clean_hook if clean_hook else "Não gerado"}

=== CONCLUSÃO (LIMPA) ===
{clean_conclusion if clean_conclusion else "Não gerada"}

=== ESTATÍSTICAS ===
Total de palavras (narração): {word_count}
Total de caracteres: {len(narration_text)}
Tempo estimado de narração: {word_count // 150} minutos
Número de capítulos: 8

=== RESUMO DA SESSÃO API ===
{json.dumps(self.api_logger.get_session_summary(), indent=2, ensure_ascii=False)}
"""
        
        # 14. Salvar arquivo -DOCS.txt
        docs_filename = f"{file_base}-DOCS.txt"
        docs_path = Path("Roteiros Prontos") / docs_filename
        with open(docs_path, 'w', encoding='utf-8') as f:
            f.write(docs_content)
        
        checklist["docs_file"] = "✅ Salvo"
        self.checklist_update.emit(title, checklist)
        self.log_message.emit(f"💾 Arquivo salvo: {docs_filename}")
        
        self.log_message.emit(f"✅ História completa gerada: {title}")
        self.log_message.emit(f"📁 Logs da API salvos em: {self.api_logger.log_file}")
        
        return docs_content
    
    def extract_chapter_from_structure(self, structure: str, chapter_num: int) -> str:
        """Extrai informações de um capítulo específico da estrutura"""
        # Procurar pelo padrão do capítulo
        patterns = [
            rf"CAPÍTULO\s+{chapter_num}[:\s\-]*(.*?)(?=CAPÍTULO\s+{chapter_num + 1}|FIM DO CAPÍTULO|$)",
            rf"Capítulo\s+{chapter_num}[:\s\-]*(.*?)(?=Capítulo\s+{chapter_num + 1}|$)",
            rf"{chapter_num}[.\s\-]*(.*?)(?={chapter_num + 1}[.\s\-]|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, structure, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0)[:1000]  # Limitar tamanho
        
        return f"Informação do capítulo {chapter_num} não encontrada na estrutura"
    
    def clean_chapter_for_narration(self, text: str) -> str:
        """Remove TODOS os marcadores de capítulo para narração limpa"""
        
        # CORREÇÃO: Regex melhorada para ser mais flexível
        # 1. re.IGNORECASE: Ignora maiúsculas/minúsculas
        # 2. \**?: Permite `**` (markdown) opcional
        # 3. \s*: Permite espaços opcionais
        # 4. (CAPÍTULO|CAPITOL|KAPITOLA): Aceita variações da palavra
        # 5. (FIM|FIN) DO: Aceita 'FIM' ou 'FIN'
        # 6. .*: Captura e remove o resto da linha (ex: ": ROĐENDANSKI ŠOK")
        
        patterns_to_remove = [
            r'^\**\s*(CAPÍTULO|CAPITOL|KAPITOLA)\s+\d+.*', # Remove "CAPÍTULO 1:", "**CAPITOL 1: ROĐENDANSKI ŠOK**"
            r'\**\s*(FIM|FIN)\s+DO\s+(CAPÍTULO|CAPITOL|KAPITOLA)\s+\d+\**.*' # Remove "FIM DO CAPÍTULO 1", "**FIM DO KAPITOLA 8**"
        ]
        
        cleaned = text
        for pattern in patterns_to_remove:
            # CORREÇÃO: Usar re.MULTILINE e re.IGNORECASE
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove linhas vazias excessivas
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def remove_all_markers(self, text: str) -> str:
        """Remove TODOS os tipos de marcadores do texto"""
        # Remove marcadores INI/END e variações
        markers = [
            r'INI\s+\w+', r'END\s+\w+',
            r'FIM\s+\w+', r'FIN\s+\w+',
            r'INÍCIO\s+\w+', r'INICIO\s+\w+',
            r'<<<.*?>>>', r'\[\[.*?\]\]'
        ]
        
        cleaned = text
        for marker in markers:
            cleaned = re.sub(marker, '', cleaned, flags=re.IGNORECASE)
        
        # Remove linhas vazias excessivas
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def send_to_gpt_with_context(self, conversation_history: list, context: dict = None) -> str:
        """Envia para GPT com contexto completo da conversa e logging"""
        try:
            # REMOVIDO: O system_message hardcoded foi removido.
            # As instruções de sistema devem vir dos arquivos .txt
            # e já estar em 'conversation_history'
            
            # A 'conversation_history' agora é enviada diretamente
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_history, # <-- CORRIGIDO
                max_tokens=4000,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            
            # Logging com limite maior para visualização
            prompt_full = json.dumps(conversation_history, ensure_ascii=False)
            self.api_logger.log_api_call(prompt_full, response_text, context)
            
            # Emitir para interface com preview maior
            # Corrigido para logar o JSON completo
            self.api_log.emit(prompt_full[:2000], response_text[:2000])
        
            return response_text
            
        except Exception as e:
            self.error.emit(f"Erro na API: {str(e)}")
            return ""
    
    def send_to_gpt(self, prompt: str, context: dict = None) -> str:
        """Envia prompt para GPT e retorna resposta com logging"""
        try:
            # REMOVIDO: system_prompt hardcoded.
            # O 'prompt' (carregado do .txt) já deve conter todas as instruções.
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    # Apenas a mensagem do usuário, que é o prompt completo do .txt
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            
            # Logging com limite maior
            self.api_logger.log_api_call(prompt, response_text, context)
            
            # Emitir para interface com preview maior
            self.api_log.emit(prompt[:2000], response_text[:2000])
            
            return response_text
            
        except Exception as e:
            self.error.emit(f"Erro na API: {str(e)}")
            return ""
    
    def generate_srt(self, text: str, filename: str):
        """Gera arquivo SRT do texto (sem marcadores)"""
        # Configurações do SRT
        DURACAO_BLOCO = 30
        INTERVALO_ENTRE_BLOCOS = 30
        MAX_CARACTERES_POR_BLOCO = 500
        MIN_PALAVRAS_POR_BLOCO = 30
        MAX_PALAVRAS_POR_BLOCO = 100
        
        # Limpar texto de qualquer marcador residual
        text = self.remove_all_markers(text)
        text = self.clean_chapter_for_narration(text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        srt_content = ""
        contador = 1
        tempo_acumulado = 0
        bloco_atual = ""
        palavras_no_bloco = 0
        palavras = text.split()
        
        i = 0
        while i < len(palavras):
            palavra = palavras[i]
            
            if (bloco_atual and 
                (len(bloco_atual) + len(palavra) + 1 > MAX_CARACTERES_POR_BLOCO or
                 palavras_no_bloco >= MAX_PALAVRAS_POR_BLOCO)):
                
                if palavras_no_bloco >= MIN_PALAVRAS_POR_BLOCO:
                    tempo_inicio = tempo_acumulado
                    tempo_fim = tempo_inicio + DURACAO_BLOCO
                    
                    srt_content += f"{contador}\n"
                    srt_content += f"{self.format_srt_time(tempo_inicio)} --> {self.format_srt_time(tempo_fim)}\n"
                    srt_content += f"{bloco_atual.strip()}\n\n"
                    
                    contador += 1
                    tempo_acumulado = tempo_fim + INTERVALO_ENTRE_BLOCOS
                    
                    bloco_atual = palavra
                    palavras_no_bloco = 1
                else:
                    bloco_atual += (" " if bloco_atual else "") + palavra
                    palavras_no_bloco += 1
            else:
                bloco_atual += (" " if bloco_atual else "") + palavra
                palavras_no_bloco += 1
            
            i += 1
        
        # Adicionar último bloco
        if bloco_atual.strip():
            tempo_inicio = tempo_acumulado
            tempo_fim = tempo_inicio + DURACAO_BLOCO
            
            srt_content += f"{contador}\n"
            srt_content += f"{self.format_srt_time(tempo_inicio)} --> {self.format_srt_time(tempo_fim)}\n"
            srt_content += f"{bloco_atual.strip()}\n\n"
        
        # Salvar arquivo SRT
        output_dir = Path("Roteiros Prontos")
        srt_path = output_dir / filename
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        self.log_message.emit(f"✅ Arquivo SRT salvo: {filename}")
    
    def format_srt_time(self, seconds: float) -> str:
        """Formata tempo para formato SRT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def load_prompt(self, path: Path, title: str) -> str:
        """Carrega e substitui variáveis no prompt"""
        try:
            # VERIFICAR SE O ARQUIVO EXISTE
            if not path.exists():
                self.error.emit(f"ERRO CRÍTICO: Arquivo não encontrado: {path}")
                self.log_message.emit(f"❌ ARQUIVO NÃO EXISTE: {path}")
                return ""
            
            # CARREGAR O ARQUIVO REAL
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # LOG DO CONTEÚDO CARREGADO
            self.log_message.emit(f"📄 Carregado arquivo: {path.name} ({len(content)} chars)")
            
            # Substituir variáveis
            content = content.replace('[TITULO]', title)
            content = content.replace('[PAIS]', self.country)
            content = content.replace('[IDIOMA]', self.language)
            
            # VERIFICAÇÃO ADICIONAL
            if len(content) < 100:
                self.log_message.emit(f"⚠️ AVISO: Arquivo muito pequeno: {path.name}")
            
            return content
            
        except Exception as e:
            self.error.emit(f"Erro ao carregar prompt {path}: {str(e)}")
            self.log_message.emit(f"❌ FALHA AO LER: {path}")
            return ""


class StoryGeneratorApp(QMainWindow):
    """Aplicação principal do gerador de histórias"""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings('StoryGenerator', 'Settings')
        self.checklist_data = {}
        self.quality_reports = {}
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        """Inicializa interface do usuário"""
        self.setWindowTitle("Gerador de Histórias Emocionantes v4.1 - CORRIGIDO")
        self.setMinimumSize(1600, 950)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Seção de API Key
        api_group = QGroupBox("Configuração da API")
        api_layout = QHBoxLayout()
        
        api_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Insira sua chave da API OpenAI (gpt-4o-mini)")
        
        self.btn_save_api = QPushButton("💾 Salvar")
        self.btn_save_api.clicked.connect(self.save_api_key)
        
        self.btn_test_api = QPushButton("🔌 Testar Conexão")
        self.btn_test_api.clicked.connect(self.test_api_connection)
        
        api_layout.addWidget(self.api_key_input)
        api_layout.addWidget(self.btn_save_api)
        api_layout.addWidget(self.btn_test_api)
        
        api_group.setLayout(api_layout)
        main_layout.addWidget(api_group)
        
        # Seção de configuração
        config_group = QGroupBox("Configurações do Roteiro")
        config_layout = QVBoxLayout()
        
        # Primeira linha - Arquivo e estilo
        row1 = QHBoxLayout()
        
        row1.addWidget(QLabel("Arquivo de Títulos:"))
        self.file_path_label = QLabel("Nenhum arquivo selecionado")
        self.file_path_label.setStyleSheet("color: #666666; font-style: italic;")
        self.btn_select_file = QPushButton("📁 Selecionar Arquivo")
        self.btn_select_file.clicked.connect(self.select_titles_file)
        
        row1.addWidget(self.file_path_label)
        row1.addWidget(self.btn_select_file)
        
        row1.addWidget(QLabel("Estilo:"))
        self.style_combo = QComboBox()
        self.style_combo.setMinimumWidth(150)
        self.load_styles()
        
        row1.addWidget(self.style_combo)
        
        config_layout.addLayout(row1)
        
        # Segunda linha - Configurações de numeração
        row2 = QHBoxLayout()
        
        row2.addWidget(QLabel("Número Inicial:"))
        self.start_num_spin = QSpinBox()
        self.start_num_spin.setRange(1, 9999)
        self.start_num_spin.setValue(156)
        self.start_num_spin.setMinimumWidth(100)
        
        row2.addWidget(self.start_num_spin)
        
        self.btn_config_siglas = QPushButton("⚙️ Configurar Siglas")
        self.btn_config_siglas.clicked.connect(self.open_config_dialog)
        
        self.btn_edit_prompts = QPushButton("📝 Editar Prompts")
        self.btn_edit_prompts.clicked.connect(self.open_prompts_editor)
        
        self.btn_view_logs = QPushButton("📊 Ver Logs API")
        self.btn_view_logs.clicked.connect(self.open_api_logs)
        
        row2.addWidget(self.btn_config_siglas)
        row2.addWidget(self.btn_edit_prompts)
        row2.addWidget(self.btn_view_logs)
        row2.addStretch()
        
        config_layout.addLayout(row2)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # Área de visualização com splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Lista de títulos (esquerda)
        titles_group = QGroupBox("Títulos Carregados")
        titles_layout = QVBoxLayout()
        
        self.titles_list = QTextEdit()
        self.titles_list.setReadOnly(True)
        self.titles_list.setMaximumWidth(400)
        self.titles_list.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
        
        self.titles_info_label = QLabel("0 títulos carregados")
        self.titles_info_label.setStyleSheet("color: #666666; font-weight: bold;")
        
        titles_layout.addWidget(self.titles_list)
        titles_layout.addWidget(self.titles_info_label)
        titles_group.setLayout(titles_layout)
        
        # Área central com tabs
        center_widget = QWidget()
        center_layout = QVBoxLayout()
        center_widget.setLayout(center_layout)
        
        # Tabs para diferentes visualizações
        self.main_tabs = QTabWidget()
        
        # Tab 1: Log e Checklist
        log_checklist_widget = QWidget()
        log_checklist_layout = QVBoxLayout()
        
        # Log de execução
        log_group = QGroupBox("Log de Execução")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                color: #333333;
                border: 1px solid #cccccc;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                line-height: 1.4;
            }
        """)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        
        # Checklist
        result_group = QGroupBox("Checklist das Histórias")
        result_layout = QVBoxLayout()
        
        self.result_tabs = QTabWidget()
        self.result_tabs.setStyleSheet("""
            QTabWidget::pane {
                background-color: #ffffff;
                border: 1px solid #cccccc;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                color: #333333;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom: 2px solid #10a37f;
            }
        """)
        
        result_layout.addWidget(self.result_tabs)
        result_group.setLayout(result_layout)
        
        log_checklist_layout.addWidget(log_group)
        log_checklist_layout.addWidget(result_group)
        log_checklist_widget.setLayout(log_checklist_layout)
        
        # Tab 2: Análise de Qualidade
        quality_widget = QWidget()
        quality_layout = QVBoxLayout()
        
        self.quality_text = QTextEdit()
        self.quality_text.setReadOnly(True)
        self.quality_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                padding: 10px;
            }
        """)
        
        quality_layout.addWidget(QLabel("📊 Relatório de Qualidade das Histórias"))
        quality_layout.addWidget(self.quality_text)
        quality_widget.setLayout(quality_layout)
        
        # Tab 3: Log da API
        api_log_widget = QWidget()
        api_log_layout = QVBoxLayout()
        
        self.api_log_text = QTextEdit()
        self.api_log_text.setReadOnly(True)
        self.api_log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                border: 1px solid #333333;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        
        api_log_layout.addWidget(QLabel("🔍 Log de Chamadas da API"))
        api_log_layout.addWidget(self.api_log_text)
        api_log_widget.setLayout(api_log_layout)
        
        # Adicionar tabs
        self.main_tabs.addTab(log_checklist_widget, "📋 Execução")
        self.main_tabs.addTab(quality_widget, "📊 Qualidade")
        self.main_tabs.addTab(api_log_widget, "🔍 API Logs")
        
        center_layout.addWidget(self.main_tabs)
        
        # Label informativa
        auto_save_label = QLabel("📁 Arquivos salvos em 'Roteiros Prontos' | 📊 Logs da API em 'API_Logs'")
        auto_save_label.setStyleSheet("color: #10a37f; font-style: italic; padding: 5px;")
        center_layout.addWidget(auto_save_label)
        
        # Adicionar ao splitter
        main_splitter.addWidget(titles_group)
        main_splitter.addWidget(center_widget)
        main_splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(main_splitter)
        
        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
                color: #333333;
                font-weight: bold;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background-color: #10a37f;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Pronto")
        self.status_label.setStyleSheet("color: #666666; padding: 5px; font-weight: bold;")
        main_layout.addWidget(self.status_label)
        
        # Botões de ação
        action_layout = QHBoxLayout()
        
        self.btn_generate = QPushButton("🚀 Gerar Histórias")
        self.btn_generate.clicked.connect(self.start_generation)
        self.btn_generate.setEnabled(False)
        
        self.btn_open_folder = QPushButton("📂 Abrir Pasta de Roteiros")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        
        self.btn_clear = QPushButton("🗑️ Limpar")
        self.btn_clear.clicked.connect(self.clear_all)
        
        action_layout.addWidget(self.btn_generate)
        action_layout.addWidget(self.btn_open_folder)
        action_layout.addWidget(self.btn_clear)
        action_layout.addStretch()
        
        main_layout.addLayout(action_layout)
        
        # Aplicar estilo geral
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #ffffff;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #333333;
            }
            QLabel {
                color: #333333;
            }
            QLineEdit, QSpinBox, QComboBox {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #ffffff;
                color: #333333;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
                border: 2px solid #10a37f;
            }
            QPushButton {
                padding: 10px 20px;
                background-color: #10a37f;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #0d8f6f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
    
    def add_to_log(self, message: str):
        """Adiciona mensagem ao log de execução"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Auto-scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def add_api_log(self, prompt: str, response: str):
        """Adiciona log de chamada da API"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Aumentar limite de exibição para 2000 caracteres
        prompt_preview = prompt[:2000] + "..." if len(prompt) > 2000 else prompt
        response_preview = response[:2000] + "..." if len(response) > 2000 else response
        
        log_entry = f"""[{timestamp}]
PROMPT ({len(prompt)} chars total):
{prompt_preview}

RESPONSE ({len(response)} chars total):
{response_preview}

{'='*50}
"""
        self.api_log_text.append(log_entry)
        scrollbar = self.api_log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_quality_report(self, title: str, analysis: dict):
        """Atualiza relatório de qualidade"""
        self.quality_reports[title] = analysis
        
        # Montar relatório visual
        report = f"""
{'='*60}
📖 {title}
{'='*60}

📊 NOTA FINAL: {analysis['score']}/10

{'✅ PONTOS FORTES:' if analysis['strengths'] else ''}
{chr(10).join(f'  • {s}' for s in analysis['strengths'])}

{'⚠️ PROBLEMAS DETECTADOS:' if analysis['issues'] else ''}
{chr(10).join(f'  • {issue}' for issue in analysis['issues'])}

{'⚡ AVISOS:' if analysis['warnings'] else ''}
{chr(10).join(f'  • {warning}' for warning in analysis['warnings'])}

📋 VERIFICAÇÕES:
  • Idioma correto: {'✅' if analysis['language_check'] else '❌'}
  • Sem repetições: {'✅' if analysis['repetition_check'] else '❌'}
  • Progressão da trama: {'✅' if analysis['plot_progression'] else '❌'}

"""
        
        # Adicionar ao texto de qualidade
        current_text = self.quality_text.toPlainText()
        self.quality_text.setPlainText(current_text + report)
    
    def update_checklist(self, title: str, checklist_data: dict):
        """Atualiza o checklist de uma história"""
        # Armazenar dados
        self.checklist_data[title] = checklist_data
        
        # Criar ou atualizar aba
        tab_index = -1
        for i in range(self.result_tabs.count()):
            if self.result_tabs.tabText(i).startswith(title[:20]):
                tab_index = i
                break
        
        # Criar widget de checklist
        checklist_widget = self.create_checklist_widget(checklist_data)
        
        if tab_index == -1:
            # Criar nova aba
            tab_title = f"{title[:20]}..." if len(title) > 20 else title
            self.result_tabs.addTab(checklist_widget, tab_title)
        else:
            # Atualizar aba existente
            self.result_tabs.removeTab(tab_index)
            self.result_tabs.insertTab(tab_index, checklist_widget, self.result_tabs.tabText(tab_index))
    
    def create_checklist_widget(self, checklist_data: dict) -> QWidget:
        """Cria widget com checklist visual"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Criar texto formatado do checklist
        checklist_text = QTextEdit()
        checklist_text.setReadOnly(True)
        
        html_content = """
        <style>
            body { font-family: Arial, sans-serif; }
            .item { margin: 5px 0; padding: 5px; }
            .pending { color: #FFA500; }
            .processing { color: #4169E1; }
            .complete { color: #10a37f; }
            .error { color: #FF0000; }
            .quality { font-weight: bold; }
        </style>
        """
        
        html_content += f"<h3>{checklist_data.get('title', 'História')}</h3>"
        html_content += "<ul>"
        
        # Items do checklist
        items = [
            ("Opções de trama", checklist_data.get('options', '⏳')),
            ("Estrutura", checklist_data.get('structure', '⏳')),
            ("Hook", checklist_data.get('hook', '⏳')),
            ("Conclusão", checklist_data.get('conclusion', '⏳')),
            ("Título traduzido", checklist_data.get('title_translated', '⏳')),
            # ("Descrição YouTube", checklist_data.get('description', '⏳')), # REMOVIDO
            ("Análise de Qualidade", checklist_data.get('quality_check', '⏳')),
            ("Arquivo NARRAR", checklist_data.get('narrar_file', '⏳')),
            ("Arquivo SRT", checklist_data.get('srt_file', '⏳')),
            ("Arquivo DOCS", checklist_data.get('docs_file', '⏳')),
        ]
        
        # Adicionar capítulos
        for i, chapter_data in enumerate(checklist_data.get('chapters', []), 1):
            status = chapter_data.get(f'chapter_{i}', '⏳')
            items.insert(2 + i, (f"Capítulo {i}", status))
        
        for label, status in items:
            css_class = "pending"
            if "✅" in status:
                css_class = "complete"
                if "Nota:" in status:
                    css_class = "quality"
            elif "🔄" in status:
                css_class = "processing"
            elif "❌" in status:
                css_class = "error"
            
            html_content += f'<li class="{css_class}">{label}: {status}</li>'
        
        html_content += "</ul>"
        
        checklist_text.setHtml(html_content)
        layout.addWidget(checklist_text)
        
        return widget
    
    def open_prompts_editor(self):
        """Abre editor de prompts"""
        style = self.style_combo.currentText()
        if not style:
            QMessageBox.warning(self, "Aviso", "Selecione um estilo primeiro")
            return
        
        prompts_dir = Path(f"Prompts/{style}")
        if not prompts_dir.exists():
            QMessageBox.warning(self, "Aviso", f"Pasta de prompts não encontrada: {prompts_dir}")
            return
        
        # Abrir pasta no explorador
        import platform
        import subprocess
        
        system = platform.system()
        if system == "Windows":
            os.startfile(str(prompts_dir))
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(prompts_dir)])
        else:  # Linux
            subprocess.run(["xdg-open", str(prompts_dir)])
    
    def open_api_logs(self):
        """Abre pasta de logs da API"""
        logs_dir = Path("API_Logs")
        logs_dir.mkdir(exist_ok=True)
        
        import platform
        import subprocess
        
        system = platform.system()
        if system == "Windows":
            os.startfile(str(logs_dir))
        elif system == "Darwin":
            subprocess.run(["open", str(logs_dir)])
        else:
            subprocess.run(["xdg-open", str(logs_dir)])
    
    def load_styles(self):
        """Carrega estilos disponíveis da pasta Prompts"""
        prompts_dir = Path("Prompts")
        if prompts_dir.exists():
            styles = [d.name for d in prompts_dir.iterdir() if d.is_dir()]
            if not styles:
                styles = ["Emocionantes"]
                self.create_default_structure()
            self.style_combo.addItems(styles)
        else:
            self.create_default_structure()
            self.style_combo.addItems(["Emocionantes"])
    
    def create_default_structure(self):
        """Cria estrutura padrão de pastas e arquivos"""
        # Criar pasta Prompts
        base_dir = Path("Prompts/Emocionantes")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar pasta Roteiros Prontos
        output_dir = Path("Roteiros Prontos")
        output_dir.mkdir(exist_ok=True)
        
        # Criar pasta API_Logs
        logs_dir = Path("API_Logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Criar arquivos de prompt padrão
        default_prompts = {
            "tenho_um_canal.txt": "# Prompt inicial\n[TITULO] [PAIS] [IDIOMA]",
            "selecionar_opcao.txt": "# Selecionar opção",
            "criar_estrutura.txt": "# Criar estrutura",
            # "criar_descricao.txt": "# Criar descrição", # REMOVIDO
            "titulo_traduzido.txt": "# Traduzir título",
            "hook.txt": "# Criar hook",
            "conclusao.txt": "# Criar conclusão",
            "texto_claude.txt": "# Prompt mestre Cap 1\n[CAPITULOS]",
            "desenvolve-caps-2a8.txt": "# Prompt loop Cap 2-8\nCHAPTER_NUM"
        }
        
        for filename, content in default_prompts.items():
            filepath = base_dir / filename
            if not filepath.exists():
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
    
    def open_output_folder(self):
        """Abre a pasta de roteiros prontos"""
        output_dir = Path("Roteiros Prontos")
        output_dir.mkdir(exist_ok=True)
        
        import platform
        import subprocess
        
        system = platform.system()
        if system == "Windows":
            os.startfile(str(output_dir))
        elif system == "Darwin":
            subprocess.run(["open", str(output_dir)])
        else:
            subprocess.run(["xdg-open", str(output_dir)])
    
    def save_api_key(self):
        """Salva a API key"""
        api_key = self.api_key_input.text().strip()
        if api_key:
            self.settings.setValue('api_key', api_key)
            QMessageBox.information(self, "Sucesso", "API Key salva com sucesso!")
            self.check_ready_to_generate()
        else:
            QMessageBox.warning(self, "Erro", "Por favor, insira uma API Key válida")
    
    def test_api_connection(self):
        """Testa conexão com a API"""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Erro", "Por favor, insira uma API Key primeiro")
            return
        
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Olá"}],
                max_tokens=10
            )
            QMessageBox.information(self, "Sucesso", "Conexão com API estabelecida com sucesso!")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao conectar com API:\n{str(e)}")
    
    def select_titles_file(self):
        """Seleciona arquivo com títulos"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar arquivo de títulos",
            "",
            "Arquivos de texto (*.txt)"
        )
        
        if file_path:
            self.load_titles_from_file(file_path)
    
    def load_titles_from_file(self, file_path: str):
        """Carrega títulos do arquivo"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                titles = [line.strip() for line in f if line.strip()]
            
            self.current_file_path = file_path
            self.current_titles = titles
            
            # Extrair informações do nome do arquivo
            filename = Path(file_path).stem
            match = re.match(r'([A-Z]+)(\d+)-(\d+)-(\d+)', filename)
            if match:
                self.file_prefix = match.group(1)
                self.file_series = match.group(2)
                self.start_num_spin.setValue(int(match.group(3)))
                self.detect_country_language(self.file_prefix)
            else:
                self.file_prefix = filename[:3].upper() if len(filename) >= 3 else "GEN"
                self.file_series = "01"
            
            # Atualizar interface
            self.file_path_label.setText(Path(file_path).name)
            self.file_path_label.setStyleSheet("color: #10a37f; font-weight: bold;")
            self.titles_list.setText("\n".join(titles))
            self.titles_info_label.setText(f"{len(titles)} títulos carregados")
            
            self.check_ready_to_generate()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar arquivo:\n{str(e)}")
    
    def detect_country_language(self, prefix: str):
        """Detecta país e idioma baseado no prefixo"""
        settings = QSettings('StoryGenerator', 'Associations')
        associations = settings.value('associations', {})
        
        if not associations:
            config_dialog = ConfigDialog(self)
            associations = config_dialog.load_associations()
        
        if prefix in associations:
            cl = associations[prefix]
            self.current_country = cl.country
            self.current_language = cl.language
            self.status_label.setText(f"✔ Detectado: {cl.country} - {cl.language}")
            self.status_label.setStyleSheet("color: #10a37f; padding: 5px; font-weight: bold;")
        else:
            self.current_country = "Brasil"
            self.current_language = "Português"
            self.status_label.setText(f"⚠ Sigla não reconhecida. Usando padrão: Brasil - Português")
            self.status_label.setStyleSheet("color: #ff9800; padding: 5px; font-weight: bold;")
    
    def check_ready_to_generate(self):
        """Verifica se está pronto para gerar"""
        api_key = self.api_key_input.text().strip()
        has_titles = hasattr(self, 'current_titles') and self.current_titles
        
        self.btn_generate.setEnabled(bool(api_key and has_titles))
    
    def open_config_dialog(self):
        """Abre diálogo de configuração de siglas"""
        dialog = ConfigDialog(self)
        if dialog.exec():
            QMessageBox.information(self, "Sucesso", "Configurações salvas!")
            if hasattr(self, 'file_prefix'):
                self.detect_country_language(self.file_prefix)
    
    def start_generation(self):
        """Inicia geração de histórias"""
        api_key = self.api_key_input.text().strip()
        style = self.style_combo.currentText()
        
        if not hasattr(self, 'current_titles'):
            QMessageBox.warning(self, "Erro", "Nenhum arquivo de títulos carregado")
            return
        
        # Limpar logs e relatórios
        self.log_text.clear()
        self.api_log_text.clear()
        self.quality_text.clear()
        self.quality_reports.clear()
        
        self.add_to_log("🚀 Iniciando geração de histórias...")
        self.add_to_log(f"🌍 País: {self.current_country}")
        self.add_to_log(f"🗣️ Idioma: {self.current_language}")
        
        # Preparar interface
        self.btn_generate.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        # Calcular total de etapas (REMOVIDA 1 ETAPA: DESCRIÇÃO)
        total_steps = len(self.current_titles) * 14 
        self.progress_bar.setMaximum(total_steps)
        self.progress_bar.setValue(0)
        
        # Criar thread
        base_name = f"{self.file_prefix}{self.file_series}"
        self.generator_thread = StoryGeneratorThread(
            api_key,
            self.current_titles,
            style,
            self.current_country,
            self.current_language,
            base_name,
            self.start_num_spin.value()
        )
        
        # Conectar sinais
        self.generator_thread.progress.connect(self.update_progress)
        self.generator_thread.log_message.connect(self.add_to_log)
        self.generator_thread.api_log.connect(self.add_api_log)
        self.generator_thread.story_generated.connect(self.add_story_tab)
        self.generator_thread.checklist_update.connect(self.update_checklist)
        self.generator_thread.quality_report.connect(self.update_quality_report)
        self.generator_thread.finished_all.connect(self.generation_complete)
        self.generator_thread.error.connect(self.handle_error)
        
        # Iniciar
        self.generator_thread.start()
    
    def update_progress(self, message: str):
        """Atualiza progresso"""
        self.status_label.setText(message)
        current = self.progress_bar.value()
        if current < self.progress_bar.maximum() - 1:
            self.progress_bar.setValue(current + 1)
    
    def add_story_tab(self, title: str, filename: str, content: str):
        """Adiciona aba com história gerada"""
        # Criar widget para mostrar o conteúdo DOCS
        text_edit = QTextEdit()
        text_edit.setPlainText(content)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #333333;
                border: none;
                padding: 15px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                line-height: 1.6;
            }
        """)
        
        # Adicionar aba DOCS
        tab_title = f"📄 {filename[:20]}..." if len(filename) > 20 else f"📄 {filename}"
        
        # Verificar se já existe uma aba com esse título
        for i in range(self.result_tabs.count()):
            if self.result_tabs.tabText(i) == tab_title:
                self.result_tabs.removeTab(i)
                break
        
        self.result_tabs.addTab(text_edit, tab_title)
        
        # Salvar conteúdo para referência
        if not hasattr(self, 'generated_stories'):
            self.generated_stories = {}
        self.generated_stories[filename] = content
    
    def generation_complete(self):
        """Chamado quando todas as histórias foram geradas"""
        self.progress_bar.setValue(self.progress_bar.maximum())
        time.sleep(0.5)
        
        self.progress_bar.setVisible(False)
        self.btn_generate.setEnabled(True)
        self.status_label.setText("✅ Geração completa! Todos os arquivos foram salvos.")
        self.status_label.setStyleSheet("color: #10a37f; padding: 5px; font-weight: bold;")
        
        self.add_to_log("✅ GERAÇÃO COMPLETA!")
        self.add_to_log(f"📁 Arquivos salvos em: Roteiros Prontos/")
        self.add_to_log(f"📊 Logs da API salvos em: API_Logs/")
        
        # Contar arquivos gerados
        output_dir = Path("Roteiros Prontos")
        narrar_files = list(output_dir.glob("*-NARRAR.txt"))
        srt_files = list(output_dir.glob("*-NARRAR.srt"))
        docs_files = list(output_dir.glob("*-DOCS.txt"))
        
        # Calcular média de qualidade
        if self.quality_reports:
            avg_score = sum(r['score'] for r in self.quality_reports.values()) / len(self.quality_reports)
            quality_summary = f"\n📊 Qualidade Média: {avg_score:.1f}/10"
        else:
            quality_summary = ""
        
        QMessageBox.information(
            self, 
            "Sucesso", 
            f"Todas as histórias foram geradas com sucesso!\n\n"
            f"📄 Arquivos NARRAR: {len(narrar_files)}\n"
            f"📄 Arquivos SRT: {len(srt_files)}\n"
            f"📄 Arquivos DOCS: {len(docs_files)}\n"
            f"{quality_summary}\n\n"
            f"Os arquivos foram salvos em:\n"
            f"📁 Roteiros Prontos/\n"
            f"📊 API_Logs/"
        )
    
    def handle_error(self, error_msg: str):
        """Trata erros durante geração"""
        self.progress_bar.setVisible(False)
        self.btn_generate.setEnabled(True)
        self.status_label.setText("❌ Erro na geração")
        self.status_label.setStyleSheet("color: #f44336; padding: 5px; font-weight: bold;")
        
        self.add_to_log(f"❌ ERRO: {error_msg}")
        
        QMessageBox.critical(self, "Erro", f"Erro durante geração:\n{error_msg}")
    
    def clear_all(self):
        """Limpa todos os resultados"""
        reply = QMessageBox.question(
            self,
            'Confirmar',
            "Limpar todas as abas e o log?\n\n"
            "(Os arquivos salvos NÃO serão apagados)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.result_tabs.clear()
            self.log_text.clear()
            self.api_log_text.clear()
            self.quality_text.clear()
            self.checklist_data.clear()
            self.quality_reports.clear()
            if hasattr(self, 'generated_stories'):
                self.generated_stories.clear()
            self.status_label.setText("Pronto")
            self.status_label.setStyleSheet("color: #666666; padding: 5px; font-weight: bold;")
            self.add_to_log("🗑️ Interface limpa")
    
    def load_settings(self):
        """Carrega configurações salvas"""
        api_key = self.settings.value('api_key', '')
        if api_key:
            self.api_key_input.setText(api_key)
    
    def closeEvent(self, event):
        """Ao fechar o aplicativo"""
        if hasattr(self, 'generator_thread') and self.generator_thread.isRunning():
            reply = QMessageBox.question(
                self,
                'Confirmar',
                'Geração em andamento. Deseja sair mesmo assim?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("StoryGenerator")
    app.setOrganizationName("StoryGenerator")
    
    # Configurar para usar tema claro
    app.setStyle('Fusion')
    
    # Paleta de cores clara
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(51, 51, 51))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Text, QColor(51, 51, 51))
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(51, 51, 51))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(16, 163, 127))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(16, 163, 127))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    
    app.setPalette(palette)
    
    window = StoryGeneratorApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
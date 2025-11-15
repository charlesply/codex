import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QProgressBar, QGroupBox, QTabWidget, QDialog, QDialogButtonBox,
    QSpinBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QPalette, QColor

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
            QPushButton {
                background-color: #10a37f;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
            }
        """)
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Sigla", "País", "Idioma"])
        self.table.horizontalHeader().setStretchLastSection(True)
        
        self.load_table()
        
        layout.addWidget(QLabel("Configurar associações:"))
        layout.addWidget(self.table)
        
        btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("➕ Adicionar")
        self.btn_add.clicked.connect(self.add_row)
        
        self.btn_remove = QPushButton("➖ Remover")
        self.btn_remove.clicked.connect(self.remove_row)
        
        self.btn_reset = QPushButton("🔄 Padrão")
        self.btn_reset.clicked.connect(self.reset_defaults)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save_and_accept)
        buttons.rejected.connect(self.reject)
        
        layout.addWidget(buttons)
        self.setLayout(layout)
        
    def load_associations(self) -> Dict[str, CountryLanguage]:
        settings = QSettings('StoryGenerator', 'Associations')
        saved = settings.value('associations', {})
        
        if not saved:
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
            }
            return defaults
        return saved
        
    def load_table(self):
        self.table.setRowCount(len(self.associations))
        for i, (code, cl) in enumerate(self.associations.items()):
            self.table.setItem(i, 0, QTableWidgetItem(code))
            self.table.setItem(i, 1, QTableWidgetItem(cl.country))
            self.table.setItem(i, 2, QTableWidgetItem(cl.language))
            
    def add_row(self):
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        
    def remove_row(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)
            
    def reset_defaults(self):
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
        
        settings = QSettings('StoryGenerator', 'Associations')
        settings.setValue('associations', new_associations)
        self.accept()


class StoryGeneratorThread(QThread):
    """Thread para gerar histórias sem travar a interface"""
    
    progress = pyqtSignal(str)
    log_message = pyqtSignal(str)
    api_log = pyqtSignal(str, str)
    story_generated = pyqtSignal(str, str, str)
    checklist_update = pyqtSignal(str, dict)
    quality_report = pyqtSignal(str, dict)
    finished_all = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, api_key: str, titles: List[str], style: str,
                 country: str, language: str, base_name: str, start_num: int,
                 max_iterations: int = 5, target_score: float = 9.0):
        super().__init__()
        self.api_key = api_key
        self.titles = titles
        self.style = style
        self.country = country
        self.language = language
        self.base_name = base_name
        self.start_num = start_num
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.client = None
        
    def run(self):
        """Executa geração de histórias"""
        try:
            self.client = OpenAI(api_key=self.api_key)
            
            output_dir = Path("Roteiros Prontos")
            output_dir.mkdir(exist_ok=True)
            
            for i, title in enumerate(self.titles):
                current_num = self.start_num + i
                file_base = f"{self.base_name}-{current_num:03d}"
                
                self.progress.emit(f"Processando história {i+1}/{len(self.titles)}: {title}")
                self.log_message.emit(f"🎬 Iniciando: {title}")
                
                result = self.generate_complete_story(title, file_base)
                
                if result:
                    self.save_files(file_base, result)
                    self.story_generated.emit(title, f"{file_base}-DOCS.txt", result['docs_content'])
                    self.log_message.emit(f"✅ História concluída: {title}")
                
            self.finished_all.emit()
            
        except Exception as e:
            self.error.emit(str(e))
    
    def generate_complete_story(self, title: str, file_base: str) -> Optional[Dict]:
        """Gera história completa com novo fluxo otimizado"""
        
        prompts_dir = Path(f"Prompts/{self.style}")
        if not prompts_dir.exists():
            self.error.emit(f"Pasta de prompts não encontrada: {prompts_dir}")
            return None
        
        checklist = self.init_checklist(title)
        
        # ETAPA 1: Gerar 3 opções de trama
        self.log_message.emit("📝 Etapa 1/7: Gerando 3 opções de trama...")
        checklist["options"] = "🔄 Processando"
        self.checklist_update.emit(title, checklist)
        
        prompt1 = self.load_prompt(prompts_dir / "01_tenho_um_canal.txt", title)
        if not prompt1:
            return None
            
        opcoes = self.send_to_gpt(prompt1)
        
        checklist["options"] = "✅ Concluído"
        self.checklist_update.emit(title, checklist)
        
        # ETAPA 2: Selecionar melhor opção
        self.log_message.emit("🎯 Etapa 2/7: Selecionando melhor opção...")
        checklist["selection"] = "🔄 Processando"
        self.checklist_update.emit(title, checklist)
        
        prompt2 = self.load_prompt(prompts_dir / "02_selecionar_opcao.txt", title)
        prompt2_full = f"{opcoes}\n\n{prompt2}"
        selecao = self.send_to_gpt(prompt2_full)
        
        checklist["selection"] = "✅ Concluído"
        self.checklist_update.emit(title, checklist)
        
        # ETAPA 3: Criar estrutura detalhada
        self.log_message.emit("📋 Etapa 3/7: Criando estrutura de 8 capítulos...")
        checklist["structure"] = "🔄 Processando"
        self.checklist_update.emit(title, checklist)
        
        prompt3 = self.load_prompt(prompts_dir / "03_criar_estrutura.txt", title)
        prompt3_full = f"{opcoes}\n\n{selecao}\n\n{prompt3}"
        estrutura = self.send_to_gpt(prompt3_full, max_tokens=2000)
        
        checklist["structure"] = "✅ Concluído"
        self.checklist_update.emit(title, checklist)
        
        # ETAPA 4: Escrever história completa
        self.log_message.emit("📖 Etapa 4/7: Escrevendo história completa (Hook + 8 caps + Conclusão)...")
        self.log_message.emit("⏱️ Isso pode levar 1-2 minutos...")
        checklist["writing"] = "🔄 Escrevendo"
        self.checklist_update.emit(title, checklist)
        
        prompt4 = self.load_prompt(prompts_dir / "04_escrever_historia_completa.txt", title)
        prompt4 = prompt4.replace('[CAPITULOS]', estrutura)
        historia_raw = self.send_to_gpt(prompt4, max_tokens=16000)
        
        # VALIDAÇÃO CRÍTICA: Verificar se história foi gerada
        word_count = len(historia_raw.split())
        
        if word_count < 100:
            self.error.emit(f"ERRO: História muito curta ({word_count} palavras). IA pode ter recusado.")
            checklist["writing"] = f"❌ Falhou ({word_count} palavras)"
            self.checklist_update.emit(title, checklist)
            return None
        
        parsed = self.parse_story_sections(historia_raw)
        
        # Verificar se tem pelo menos 3 capítulos parseados
        caps_encontrados = sum(1 for i in range(1, 9) if parsed.get(f'cap{i}', '').strip())
        
        if caps_encontrados < 3:
            self.error.emit(f"ERRO: Apenas {caps_encontrados} capítulos encontrados. Formato inválido.")
            checklist["writing"] = f"❌ Formato inválido"
            self.checklist_update.emit(title, checklist)
            return None
        
        self.log_message.emit(f"✅ História escrita: {word_count} palavras ({caps_encontrados} caps)")
        checklist["writing"] = f"✅ {word_count} palavras"
        self.checklist_update.emit(title, checklist)
        
        # ETAPAS 5-7: Análise, Correção e Polish
        self.log_message.emit("🔍 Etapas 5-7/7: Análise e revisão automática...")
        checklist["quality"] = "🔄 Analisando"
        self.checklist_update.emit(title, checklist)
        
        final_result = self.analyze_and_fix_loop(estrutura, parsed['full_story'], title, checklist)
        
        # Montar resultado final
        result = {
            'title': title,
            'estrutura': estrutura,
            'historia_completa': final_result['story'],
            'analise': final_result['analysis'],
            'iterations': final_result['iterations'],
            'final_score': final_result['final_score'],
            'parsed': self.parse_story_sections(final_result['story']),
            'docs_content': self.create_docs_content(title, estrutura, final_result)
        }
        
        return result
    
    def analyze_and_fix_loop(self, estrutura: str, historia: str, 
                            title: str, checklist: dict) -> Dict:
        """Loop de análise e correção até atingir score >= 9.0"""
        
        prompts_dir = Path(f"Prompts/{self.style}")
        current_story = historia
        best_story = historia
        best_score = 0
        iteration = 0
        
        while iteration < self.max_iterations:
            # ANÁLISE
            self.log_message.emit(f"🔍 Análise {iteration + 1}/{self.max_iterations}...")
            
            prompt_analise = self.load_prompt(prompts_dir / "05_analisar_qualidade.txt", title)
            prompt_analise = prompt_analise.replace('[ESTRUTURA_PLANEJADA]', estrutura)
            prompt_analise = prompt_analise.replace('[HISTORIA_COMPLETA]', current_story)
            
            analise_raw = self.send_to_gpt(prompt_analise, max_tokens=2000)
            
            # Parse JSON
            try:
                analise = self.extract_json_from_response(analise_raw)
            except Exception as e:
                self.log_message.emit(f"⚠️ Erro ao parsear análise: {e}")
                self.log_message.emit(f"Resposta bruta: {analise_raw[:500]}")
                analise = {
                    'score_geral': 8.0,
                    'problemas_criticos': [],
                    'scores_detalhados': {},
                    'pontos_fortes': []
                }
            
            score = float(analise.get('score_geral', 8.0))
            self.log_message.emit(f"📊 Score: {score}/10")
            
            # Atualizar melhor versão
            if score > best_score:
                best_score = score
                best_story = current_story
            
            # Emitir relatório
            self.quality_report.emit(title, analise)
            
            checklist["quality"] = f"📊 {score}/10 (Tent. {iteration + 1}/{self.max_iterations})"
            self.checklist_update.emit(title, checklist)
            
            # Atingiu meta?
            if score >= self.target_score:
                self.log_message.emit(f"🎉 Meta atingida! Score: {score}/10")
                
                # Polish final
                self.log_message.emit("✨ Aplicando polish final...")
                prompt_polish = self.load_prompt(prompts_dir / "07_polish_final.txt", title)
                prompt_polish = prompt_polish.replace('[SCORE]', str(score))
                prompt_polish = prompt_polish.replace('[HISTORIA_COMPLETA]', current_story)
                
                polished = self.send_to_gpt(prompt_polish, max_tokens=16000)
                
                checklist["quality"] = f"✅ Aprovado: {score}/10"
                self.checklist_update.emit(title, checklist)
                
                return {
                    'story': polished,
                    'analysis': analise,
                    'iterations': iteration + 1,
                    'final_score': score
                }
            
            # Precisa corrigir
            problemas = analise.get('problemas_criticos', [])
            if not problemas:
                self.log_message.emit(f"⚠️ Score {score}/10 mas sem problemas listados")
                break
            
            self.log_message.emit(f"🔧 Corrigindo {len(problemas)} problemas...")
            
            # CORREÇÃO
            prompt_correcao = self.load_prompt(prompts_dir / "06_corrigir_problemas.txt", title)
            prompt_correcao = prompt_correcao.replace('[ESTRUTURA_PLANEJADA]', estrutura)
            prompt_correcao = prompt_correcao.replace('[HISTORIA_COM_PROBLEMAS]', current_story)
            prompt_correcao = prompt_correcao.replace('[SCORE]', str(score))
            prompt_correcao = prompt_correcao.replace('[PROBLEMAS_JSON]', 
                json.dumps(problemas, indent=2, ensure_ascii=False))
            prompt_correcao = prompt_correcao.replace('[SUGESTOES_JSON]',
                json.dumps(analise.get('sugestoes_correcao', []), indent=2, ensure_ascii=False))
            
            current_story = self.send_to_gpt(prompt_correcao, max_tokens=16000)
            
            iteration += 1
        
        # Não atingiu meta
        self.log_message.emit(f"⚠️ Limite de iterações atingido. Melhor score: {best_score}/10")
        
        checklist["quality"] = f"⚠️ Score final: {best_score}/10"
        self.checklist_update.emit(title, checklist)
        
        return {
            'story': best_story,
            'analysis': analise,
            'iterations': self.max_iterations,
            'final_score': best_score
        }
    
    def extract_json_from_response(self, text: str) -> dict:
        """Extrai JSON da resposta da API"""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        
        # Tentar encontrar o objeto JSON
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            return json.loads(json_str)
        else:
            return json.loads(text)
    
    def parse_story_sections(self, story_text: str) -> dict:
        """Parse das seções da história"""
        sections = {}
        
        patterns = {
            'hook': r'===\s*HOOK\s*===(.*?)(?====|\Z)',
            'cap1': r'===\s*CAPÍTULO 1\s*===(.*?)(?====|\Z)',
            'cap2': r'===\s*CAPÍTULO 2\s*===(.*?)(?====|\Z)',
            'cap3': r'===\s*CAPÍTULO 3\s*===(.*?)(?====|\Z)',
            'cap4': r'===\s*CAPÍTULO 4\s*===(.*?)(?====|\Z)',
            'cap5': r'===\s*CAPÍTULO 5\s*===(.*?)(?====|\Z)',
            'cap6': r'===\s*CAPÍTULO 6\s*===(.*?)(?====|\Z)',
            'cap7': r'===\s*CAPÍTULO 7\s*===(.*?)(?====|\Z)',
            'cap8': r'===\s*CAPÍTULO 8\s*===(.*?)(?====|\Z)',
            'conclusao': r'===\s*CONCLUSÃO\s*===(.*?)(?====|\Z)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, story_text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()
            else:
                sections[key] = ""
        
        # História completa limpa
        full_clean = story_text
        for marker in ['=== HOOK ===', '=== CAPÍTULO', '=== CONCLUSÃO ===']:
            full_clean = re.sub(r'===\s*' + re.escape(marker.replace('===', '').strip()) + r'.*?===', '', full_clean, flags=re.IGNORECASE)
        
        sections['full_story'] = full_clean.strip()
        
        return sections
    
    def create_docs_content(self, title: str, estrutura: str, result: dict) -> str:
        """Cria conteúdo do arquivo DOCS"""
        analise = result['analysis']
        
        docs = f"""=== INFORMAÇÕES DO ROTEIRO ===
Título: {title}
País: {self.country}
Idioma: {self.language}
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}

=== ANÁLISE DE QUALIDADE ===
Score Final: {result['final_score']}/10
Iterações de Correção: {result['iterations']}

Scores Detalhados:
{json.dumps(analise.get('scores_detalhados', {}), indent=2, ensure_ascii=False)}

Problemas Críticos: {len(analise.get('problemas_criticos', []))}
Problemas Menores: {len(analise.get('problemas_menores', []))}

Pontos Fortes:
{chr(10).join('• ' + p for p in analise.get('pontos_fortes', []))}

{f"Problemas Encontrados:{chr(10)}{chr(10).join('• ' + p.get('descricao', '') for p in analise.get('problemas_criticos', []))}" if analise.get('problemas_criticos') else ""}

=== ESTRUTURA DOS CAPÍTULOS ===
{estrutura}

=== ESTATÍSTICAS ===
Total de palavras: {len(result['story'].split())}
Total de caracteres: {len(result['story'])}
Tempo estimado de narração: {len(result['story'].split()) // 150} minutos

=== HISTÓRIA COMPLETA (COM DELIMITADORES) ===
{result['story']}
"""
        return docs
    
    def save_files(self, file_base: str, result: dict):
        """Salva todos os arquivos"""
        output_dir = Path("Roteiros Prontos")
        parsed = result['parsed']
        
        # NARRAR.txt
        narrar_text = ""
        if parsed['hook']:
            narrar_text += parsed['hook'] + "\n\n"
        for i in range(1, 9):
            cap = parsed.get(f'cap{i}', '')
            if cap:
                narrar_text += cap + "\n\n"
        if parsed['conclusao']:
            narrar_text += parsed['conclusao']
        
        narrar_file = output_dir / f"{file_base}-NARRAR.txt"
        with open(narrar_file, 'w', encoding='utf-8') as f:
            f.write(narrar_text.strip())
        
        self.log_message.emit(f"💾 Salvo: {file_base}-NARRAR.txt")
        
        # SRT
        self.generate_srt(narrar_text, f"{file_base}-NARRAR.srt")
        
        # DOCS
        docs_file = output_dir / f"{file_base}-DOCS.txt"
        with open(docs_file, 'w', encoding='utf-8') as f:
            f.write(result['docs_content'])
        
        self.log_message.emit(f"💾 Salvo: {file_base}-DOCS.txt")
    
    def generate_srt(self, text: str, filename: str):
        """Gera arquivo SRT"""
        DURACAO = 30
        INTERVALO = 30
        MAX_CHARS = 500
        
        output_dir = Path("Roteiros Prontos")
        srt_content = ""
        contador = 1
        tempo = 0
        
        palavras = text.split()
        bloco = ""
        
        for palavra in palavras:
            if len(bloco) + len(palavra) + 1 > MAX_CHARS and bloco:
                inicio = tempo
                fim = inicio + DURACAO
                
                srt_content += f"{contador}\n"
                srt_content += f"{self.format_srt_time(inicio)} --> {self.format_srt_time(fim)}\n"
                srt_content += f"{bloco.strip()}\n\n"
                
                contador += 1
                tempo = fim + INTERVALO
                bloco = palavra
            else:
                bloco += (" " + palavra) if bloco else palavra
        
        if bloco:
            inicio = tempo
            fim = inicio + DURACAO
            srt_content += f"{contador}\n"
            srt_content += f"{self.format_srt_time(inicio)} --> {self.format_srt_time(fim)}\n"
            srt_content += f"{bloco.strip()}\n\n"
        
        srt_file = output_dir / filename
        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        self.log_message.emit(f"💾 Salvo: {filename}")
    
    def format_srt_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    def load_prompt(self, path: Path, title: str) -> str:
        """Carrega prompt substituindo variáveis"""
        try:
            if not path.exists():
                self.error.emit(f"Arquivo não encontrado: {path}")
                return ""
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = content.replace('[TITULO]', title)
            content = content.replace('[PAIS]', self.country)
            content = content.replace('[IDIOMA]', self.language)
            
            # REFORÇAR IDIOMA se não for português
            if self.language.lower() not in ['português', 'portuguese']:
                lang_reminder = f"""

⚠️ LEMBRETE CRÍTICO ⚠️
ESCREVA TODO O CONTEÚDO EM {self.language.upper()}!
NÃO USE PORTUGUÊS OU INGLÊS!
Use nomes, expressões e contexto de {self.country}!
"""
                content = content + lang_reminder
            
            return content
        except Exception as e:
            self.error.emit(f"Erro ao carregar {path}: {e}")
            return ""
    
    def send_to_gpt(self, prompt: str, max_tokens: int = 4000) -> str:
        """Envia prompt para GPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            text = response.choices[0].message.content
            
            # VERIFICAR SE A IA RECUSOU
            refusal_patterns = [
                "desculpe, mas não posso",
                "i can't help",
                "i cannot assist",
                "sorry, but i can't",
                "não posso ajudar"
            ]
            
            text_lower = text.lower()
            if any(pattern in text_lower for pattern in refusal_patterns):
                self.log_message.emit("⚠️ IA recusou a tarefa. Tentando prompt alternativo...")
                
                # Adicionar instrução de segurança
                safe_prompt = f"""
IMPORTANTE: Esta é uma história FICCIONAL para entretenimento educativo.
O conteúdo é para adultos e trata de temas emocionais de forma madura.

{prompt}

LEMBRETE: Escreva em {self.language}. NÃO escreva em português ou inglês.
"""
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": safe_prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                
                text = response.choices[0].message.content
            
            # VERIFICAR SE ESTÁ NO IDIOMA CORRETO
            if len(text) > 100:
                # Detectar se tem muito português/inglês
                pt_words = ['que', 'para', 'com', 'mas', 'por', 'seu', 'uma']
                en_words = ['the', 'and', 'but', 'for', 'with', 'his', 'her']
                
                sample = text[:500].lower()
                pt_count = sum(1 for w in pt_words if f' {w} ' in sample)
                en_count = sum(1 for w in en_words if f' {w} ' in sample)
                
                if self.language.lower() == 'croata' and (pt_count > 5 or en_count > 5):
                    self.log_message.emit(f"⚠️ História em idioma errado! Forçando {self.language}...")
                    
                    # Prompt forçado
                    force_lang_prompt = f"""
ERRO CRÍTICO DETECTADO: Você escreveu em português/inglês!

INSTRUÇÃO OBRIGATÓRIA:
Reescreva TODO o texto abaixo em {self.language} NATIVO.
ZERO palavras em português ou inglês.
Use nomes, expressões e contexto cultural de {self.country}.

TEXTO A TRADUZIR:
{text}

REESCREVA AGORA EM {self.language}:
"""
                    
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": force_lang_prompt}],
                        max_tokens=max_tokens,
                        temperature=0.7
                    )
                    
                    text = response.choices[0].message.content
            
            # Log resumido
            self.api_log.emit(prompt[:1000], text[:1000])
            
            return text
        except Exception as e:
            self.error.emit(f"Erro na API: {e}")
            return ""
    
    def init_checklist(self, title: str) -> dict:
        return {
            "title": title,
            "options": "⏳ Aguardando",
            "selection": "⏳ Aguardando",
            "structure": "⏳ Aguardando",
            "writing": "⏳ Aguardando",
            "quality": "⏳ Aguardando",
        }


class StoryGeneratorApp(QMainWindow):
    """Aplicação principal"""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings('StoryGenerator', 'Settings')
        self.checklist_data = {}
        self.quality_reports = {}
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        self.setWindowTitle("Gerador de Histórias v5.0 - Refatorado")
        self.setMinimumSize(1600, 950)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Info version
        info_label = QLabel("✨ VERSÃO 5.0: Escrita completa em UMA chamada + Auto-revisão até 5x")
        info_label.setStyleSheet("color: #10a37f; font-weight: bold; padding: 10px; font-size: 12px;")
        main_layout.addWidget(info_label)
        
        # API Key
        api_group = QGroupBox("Configuração da API")
        api_layout = QHBoxLayout()
        
        api_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Insira sua chave da API OpenAI")
        
        self.btn_save_api = QPushButton("💾 Salvar")
        self.btn_save_api.clicked.connect(self.save_api_key)
        
        self.btn_test_api = QPushButton("🔌 Testar")
        self.btn_test_api.clicked.connect(self.test_api_connection)
        
        api_layout.addWidget(self.api_key_input)
        api_layout.addWidget(self.btn_save_api)
        api_layout.addWidget(self.btn_test_api)
        
        api_group.setLayout(api_layout)
        main_layout.addWidget(api_group)
        
        # Configurações
        config_group = QGroupBox("Configurações do Roteiro")
        config_layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Arquivo:"))
        self.file_path_label = QLabel("Nenhum arquivo selecionado")
        self.file_path_label.setStyleSheet("color: #666; font-style: italic;")
        self.btn_select_file = QPushButton("📁 Selecionar")
        self.btn_select_file.clicked.connect(self.select_titles_file)
        
        row1.addWidget(self.file_path_label)
        row1.addWidget(self.btn_select_file)
        
        row1.addWidget(QLabel("Estilo:"))
        self.style_combo = QComboBox()
        self.style_combo.setMinimumWidth(150)
        self.load_styles()
        row1.addWidget(self.style_combo)
        
        config_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Número Inicial:"))
        self.start_num_spin = QSpinBox()
        self.start_num_spin.setRange(1, 9999)
        self.start_num_spin.setValue(1)
        row2.addWidget(self.start_num_spin)
        
        self.btn_config_siglas = QPushButton("⚙️ Configurar Siglas")
        self.btn_config_siglas.clicked.connect(self.open_config_dialog)
        row2.addWidget(self.btn_config_siglas)
        
        row2.addStretch()
        config_layout.addLayout(row2)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Títulos
        titles_group = QGroupBox("Títulos Carregados")
        titles_layout = QVBoxLayout()
        
        self.titles_list = QTextEdit()
        self.titles_list.setReadOnly(True)
        self.titles_list.setMaximumWidth(400)
        
        self.titles_info_label = QLabel("0 títulos carregados")
        
        titles_layout.addWidget(self.titles_list)
        titles_layout.addWidget(self.titles_info_label)
        titles_group.setLayout(titles_layout)
        
        # Tabs
        self.main_tabs = QTabWidget()
        
        # Tab 1: Log
        log_widget = QWidget()
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(QLabel("📋 Log de Execução"))
        log_layout.addWidget(self.log_text)
        log_widget.setLayout(log_layout)
        
        # Tab 2: Qualidade
        quality_widget = QWidget()
        quality_layout = QVBoxLayout()
        
        self.quality_text = QTextEdit()
        self.quality_text.setReadOnly(True)
        
        quality_layout.addWidget(QLabel("📊 Relatório de Qualidade"))
        quality_layout.addWidget(self.quality_text)
        quality_widget.setLayout(quality_layout)
        
        # Tab 3: API Log
        api_log_widget = QWidget()
        api_log_layout = QVBoxLayout()
        
        self.api_log_text = QTextEdit()
        self.api_log_text.setReadOnly(True)
        self.api_log_text.setStyleSheet("background: #1e1e1e; color: #00ff00; font-family: monospace;")
        
        api_log_layout.addWidget(QLabel("🔍 Log da API"))
        api_log_layout.addWidget(self.api_log_text)
        api_log_widget.setLayout(api_log_layout)
        
        self.main_tabs.addTab(log_widget, "📋 Execução")
        self.main_tabs.addTab(quality_widget, "📊 Qualidade")
        self.main_tabs.addTab(api_log_widget, "🔍 API")
        
        main_splitter.addWidget(titles_group)
        main_splitter.addWidget(self.main_tabs)
        main_splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(main_splitter)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Pronto")
        self.status_label.setStyleSheet("color: #666; padding: 5px;")
        main_layout.addWidget(self.status_label)
        
        # Botões
        action_layout = QHBoxLayout()
        
        self.btn_generate = QPushButton("🚀 Gerar Histórias (Novo Fluxo)")
        self.btn_generate.clicked.connect(self.start_generation)
        self.btn_generate.setEnabled(False)
        
        self.btn_open_folder = QPushButton("📂 Abrir Pasta")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        
        self.btn_clear = QPushButton("🗑️ Limpar")
        self.btn_clear.clicked.connect(self.clear_all)
        
        action_layout.addWidget(self.btn_generate)
        action_layout.addWidget(self.btn_open_folder)
        action_layout.addWidget(self.btn_clear)
        action_layout.addStretch()
        
        main_layout.addLayout(action_layout)
        
        # Aplicar estilo
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #f5f5f5; 
            }
            QLabel {
                color: #333333;
                background-color: transparent;
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
                color: #333333;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit, QSpinBox, QComboBox {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #ffffff;
                color: #333333;
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
            QTextEdit { 
                background-color: #ffffff; 
                border: 1px solid #cccccc;
                color: #333333;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: #ffffff;
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
    
    def load_styles(self):
        prompts_dir = Path("Prompts")
        if prompts_dir.exists():
            styles = [d.name for d in prompts_dir.iterdir() if d.is_dir()]
            if styles:
                self.style_combo.addItems(styles)
            else:
                self.style_combo.addItems(["Emocionantes"])
        else:
            prompts_dir.mkdir()
            (prompts_dir / "Emocionantes").mkdir()
            self.style_combo.addItems(["Emocionantes"])
    
    def save_api_key(self):
        api_key = self.api_key_input.text().strip()
        if api_key:
            self.settings.setValue('api_key', api_key)
            QMessageBox.information(self, "Sucesso", "API Key salva!")
            self.check_ready_to_generate()
        else:
            QMessageBox.warning(self, "Erro", "Insira uma API Key válida")
    
    def test_api_connection(self):
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Erro", "Insira uma API Key primeiro")
            return
        
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Olá"}],
                max_tokens=10
            )
            QMessageBox.information(self, "Sucesso", "Conexão com API OK!")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro:\n{str(e)}")
    
    def select_titles_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar arquivo", "", "Arquivos de texto (*.txt)"
        )
        
        if file_path:
            self.load_titles_from_file(file_path)
    
    def load_titles_from_file(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                titles = [line.strip() for line in f if line.strip()]
            
            self.current_file_path = file_path
            self.current_titles = titles
            
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
            
            self.file_path_label.setText(Path(file_path).name)
            self.file_path_label.setStyleSheet("color: #10a37f; font-weight: bold;")
            self.titles_list.setText("\n".join(titles))
            self.titles_info_label.setText(f"{len(titles)} títulos carregados")
            
            self.check_ready_to_generate()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar:\n{e}")
    
    def detect_country_language(self, prefix: str):
        settings = QSettings('StoryGenerator', 'Associations')
        associations = settings.value('associations', {})
        
        if not associations:
            config_dialog = ConfigDialog(self)
            associations = config_dialog.load_associations()
        
        if prefix in associations:
            cl = associations[prefix]
            self.current_country = cl.country
            self.current_language = cl.language
            self.status_label.setText(f"✓ {cl.country} - {cl.language}")
            self.status_label.setStyleSheet("color: #10a37f; padding: 5px;")
        else:
            self.current_country = "Brasil"
            self.current_language = "Português"
            self.status_label.setText(f"⚠ Padrão: Brasil - Português")
    
    def check_ready_to_generate(self):
        api_key = self.api_key_input.text().strip()
        has_titles = hasattr(self, 'current_titles') and self.current_titles
        self.btn_generate.setEnabled(bool(api_key and has_titles))
    
    def open_config_dialog(self):
        dialog = ConfigDialog(self)
        if dialog.exec():
            QMessageBox.information(self, "Sucesso", "Configurações salvas!")
            if hasattr(self, 'file_prefix'):
                self.detect_country_language(self.file_prefix)
    
    def start_generation(self):
        api_key = self.api_key_input.text().strip()
        style = self.style_combo.currentText()
        
        if not hasattr(self, 'current_titles'):
            QMessageBox.warning(self, "Erro", "Carregue um arquivo de títulos")
            return
        
        self.log_text.clear()
        self.api_log_text.clear()
        self.quality_text.clear()
        
        self.log_text.append("🚀 Iniciando geração...")
        self.log_text.append(f"🌍 País: {self.current_country}")
        self.log_text.append(f"🗣️ Idioma: {self.current_language}")
        self.log_text.append(f"📚 Estilo: {style}")
        
        self.btn_generate.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.current_titles) * 7)
        self.progress_bar.setValue(0)
        
        base_name = f"{self.file_prefix}{self.file_series}"
        
        self.generator_thread = StoryGeneratorThread(
            api_key,
            self.current_titles,
            style,
            self.current_country,
            self.current_language,
            base_name,
            self.start_num_spin.value(),
            max_iterations=5,
            target_score=9.0
        )
        
        self.generator_thread.progress.connect(self.update_progress)
        self.generator_thread.log_message.connect(self.add_to_log)
        self.generator_thread.api_log.connect(self.add_api_log)
        self.generator_thread.quality_report.connect(self.update_quality_report)
        self.generator_thread.finished_all.connect(self.generation_complete)
        self.generator_thread.error.connect(self.handle_error)
        
        self.generator_thread.start()
    
    def update_progress(self, message: str):
        self.status_label.setText(message)
        current = self.progress_bar.value()
        if current < self.progress_bar.maximum():
            self.progress_bar.setValue(current + 1)
    
    def add_to_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def add_api_log(self, prompt: str, response: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"""[{timestamp}]
PROMPT ({len(prompt)} chars):
{prompt}

RESPONSE ({len(response)} chars):
{response}

{'='*50}
"""
        self.api_log_text.append(log_entry)
    
    def update_quality_report(self, title: str, analysis: dict):
        self.quality_reports[title] = analysis
        
        report = f"""
{'='*60}
📖 {title}
{'='*60}

📊 SCORE: {analysis.get('score_geral', 'N/A')}/10

Scores Detalhados:
{json.dumps(analysis.get('scores_detalhados', {}), indent=2, ensure_ascii=False)}

✅ Pontos Fortes:
{chr(10).join('  • ' + s for s in analysis.get('pontos_fortes', []))}

{'⚠️ Problemas:' if analysis.get('problemas_criticos') else ''}
{chr(10).join('  • ' + p.get('descricao', '') for p in analysis.get('problemas_criticos', []))}

"""
        self.quality_text.append(report)
    
    def generation_complete(self):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_bar.setVisible(False)
        self.btn_generate.setEnabled(True)
        self.status_label.setText("✅ Geração completa!")
        
        self.add_to_log("✅ TODAS AS HISTÓRIAS GERADAS!")
        
        QMessageBox.information(
            self, "Sucesso",
            f"Histórias geradas com sucesso!\n\n"
            f"Arquivos salvos em: Roteiros Prontos/"
        )
    
    def handle_error(self, error_msg: str):
        self.progress_bar.setVisible(False)
        self.btn_generate.setEnabled(True)
        self.status_label.setText("❌ Erro")
        
        self.add_to_log(f"❌ ERRO: {error_msg}")
        
        QMessageBox.critical(self, "Erro", f"Erro:\n{error_msg}")
    
    def clear_all(self):
        reply = QMessageBox.question(
            self, 'Confirmar',
            "Limpar logs?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.log_text.clear()
            self.api_log_text.clear()
            self.quality_text.clear()
            self.status_label.setText("Pronto")
    
    def open_output_folder(self):
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
    
    def load_settings(self):
        api_key = self.settings.value('api_key', '')
        if api_key:
            self.api_key_input.setText(api_key)
            self.check_ready_to_generate()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("StoryGenerator")
    app.setOrganizationName("StoryGenerator")
    
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(51, 51, 51))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = StoryGeneratorApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
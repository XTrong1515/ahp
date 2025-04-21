from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Sử dụng Agg backend cho matplotlib
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import seaborn as sns


app = Flask(__name__)

# Cấu hình kết nối PostgreSQL
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:123@localhost:5432/recruitment_ahp'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#Cấu hình kết nối MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Theanh412%40@localhost:3306/recruitment_ahp'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Khởi tạo SQLAlchemy
db = SQLAlchemy(app)

# Định nghĩa các model
class RecruitmentRound(db.Model):
    __tablename__ = 'recruitment_rounds'
    round_id = db.Column(db.Integer, primary_key=True)
    round_name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    position = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class RecruitmentCriteria(db.Model):
    __tablename__ = 'recruitment_criteria'
    criterion_id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey('recruitment_rounds.round_id'), nullable=False)
    criterion_name = db.Column(db.String(100), nullable=False)
    is_custom = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class CriteriaMatrix(db.Model):
    __tablename__ = 'criteria_matrix'
    matrix_id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey('recruitment_rounds.round_id'), nullable=False)
    matrix_data = db.Column(db.JSON, nullable=False)  # Sử dụng JSON
    consistency_ratio = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class Candidate(db.Model):
    __tablename__ = 'candidates'
    candidate_id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey('recruitment_rounds.round_id'), nullable=False)
    full_name = db.Column(db.String(255), nullable=False)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class CandidateScore(db.Model):
    __tablename__ = 'candidate_scores'
    score_id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey('recruitment_rounds.round_id'), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidates.candidate_id'), nullable=False)
    total_score = db.Column(db.Float, nullable=False)
    ranking = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# Tiêu chí mặc định
criteria = ["Kiến thức chuyên môn", "Kinh nghiệm", "Kỹ năng mềm", "Tinh thần trách nhiệm", "Mức lương mong muốn", "Phù hợp với văn hóa"]
num_criteria = len(criteria)

default_pairwise = np.array([
    [1, 2, 3, 3, 5, 4],
    [1/2, 1, 2, 2, 4, 3],
    [1/3, 1/2, 1, 2, 3, 2],
    [1/3, 1/2, 1/2, 1, 2, 2],
    [1/5, 1/4, 1/3, 1/2, 1, 1/2],
    [1/4, 1/3, 1/2, 1/2, 2, 1]
])

# Giá trị tối ưu cho từng tiêu chí
optimal_values = np.array([100, 10, 10, 10, 5, 10])

# Kiểm tra kết nối đến database
@app.route('/check_connection')
def check_connection():
    try:
        RecruitmentRound.query.first()
        return "Database connection is successful."
    except Exception as e:
        return f"Database connection failed: {str(e)}"

# Trang chính: Tạo đợt tuyển dụng và nhập thông tin
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Lấy thông tin đợt tuyển dụng
        round_name = request.form['round_name']
        description = request.form['description']
        position = request.form['position']
        num_candidates = int(request.form['num_candidates'])

        # Tạo đợt tuyển dụng mới
        try:
            new_round = RecruitmentRound(
                round_name=round_name,
                description=description,
                position=position
            )
            db.session.add(new_round)
            db.session.commit()

            # Lưu tiêu chí
            for criterion_name in criteria:
                criterion = RecruitmentCriteria(
                    round_id=new_round.round_id,
                    criterion_name=criterion_name
                )
                db.session.add(criterion)
            db.session.commit()

            # Chuyển numpy array sang list để sử dụng trong template
            default_pairwise_list = default_pairwise.tolist() if isinstance(default_pairwise, np.ndarray) else default_pairwise
            optimal_values_list = optimal_values.tolist() if isinstance(optimal_values, np.ndarray) else optimal_values

            return render_template('index.html', 
                                  round_id=new_round.round_id, 
                                  num_candidates=num_candidates, 
                                  criteria=criteria, 
                                  pairwise=default_pairwise_list, 
                                  optimal_values=optimal_values_list)

        except Exception as e:
            db.session.rollback()
            return f"Lỗi khi tạo đợt tuyển dụng: {str(e)}"

    # Chuyển numpy array sang list để sử dụng trong template
    default_pairwise_list = default_pairwise.tolist() if isinstance(default_pairwise, np.ndarray) else default_pairwise
    optimal_values_list = optimal_values.tolist() if isinstance(optimal_values, np.ndarray) else optimal_values

    return render_template('index.html', 
                          num_candidates=0, 
                          criteria=criteria, 
                          pairwise=default_pairwise_list, 
                          optimal_values=optimal_values_list)

# Nhập thông tin ứng viên và so sánh tiêu chí
def create_criterion_weights_chart(criteria, weights):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(criteria, weights, color='skyblue')
    plt.xlabel('Tiêu chí', fontsize=12)
    plt.ylabel('Trọng số', fontsize=12)
    plt.title('Trọng số các tiêu chí', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, max(weights) * 1.2)  # Điều chỉnh giới hạn trục y để có khoảng trắng

    # Thêm giá trị trọng số trên mỗi cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Chuyển biểu đồ thành chuỗi base64 để hiển thị trong HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded_img

def create_candidate_scores_chart(scores, candidate_names):
    """Tạo biểu đồ điểm số của các ứng viên"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(candidate_names, scores, color='lightcoral')
    plt.xlabel('Ứng viên', fontsize=12)
    plt.ylabel('Tổng điểm', fontsize=12)
    plt.title('Điểm số tổng của các ứng viên', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, max(scores) * 1.2)  # Điều chỉnh giới hạn trục y

    # Thêm giá trị điểm số trên mỗi cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded_img

def create_criteria_comparison_chart(candidate_names, criteria, candidate_values, weights):
    """Tạo biểu đồ so sánh điểm của ứng viên theo từng tiêu chí"""
    num_candidates = len(candidate_names)
    num_criteria = len(criteria)
    bar_width = 0.8 / num_candidates
    index = np.arange(num_criteria)

    plt.figure(figsize=(12, 7))

    for i, name in enumerate(candidate_names):
        values = candidate_values[i]
        # Nhân giá trị với trọng số để thể hiện mức độ quan trọng của tiêu chí
        weighted_values = values * weights
        plt.bar(index + i * bar_width, weighted_values, bar_width, label=name)

    plt.xlabel('Tiêu chí', fontsize=12)
    plt.ylabel('Điểm số (đã nhân trọng số)', fontsize=12)
    plt.title('So sánh điểm của ứng viên theo từng tiêu chí (đã nhân trọng số)', fontsize=14, fontweight='bold')
    plt.xticks(index + bar_width * (num_candidates - 1) / 2, criteria, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded_img

def create_raw_data_chart(candidate_names, criteria, candidate_values):
    """Tạo biểu đồ hiển thị dữ liệu thô của ứng viên theo từng tiêu chí"""
    num_candidates = len(candidate_names)
    num_criteria = len(criteria)
    bar_width = 0.8 / num_candidates
    index = np.arange(num_criteria)

    plt.figure(figsize=(12, 7))

    for i, name in enumerate(candidate_names):
        values = candidate_values[i]
        plt.bar(index + i * bar_width, values, bar_width, label=name)

    plt.xlabel('Tiêu chí', fontsize=12)
    plt.ylabel('Điểm số gốc', fontsize=12)
    plt.title('Điểm số gốc của ứng viên theo từng tiêu chí', fontsize=14, fontweight='bold')
    plt.xticks(index + bar_width * (num_candidates - 1) / 2, criteria, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded_img

def create_pairwise_matrix_visualization(pairwise_matrix, criteria):
    """Tạo heatmap để hiển thị ma trận so sánh cặp"""
    plt.figure(figsize=(10, 8))
    
    # Tạo heatmap
    sns.heatmap(pairwise_matrix, annot=True, cmap="YlGnBu", fmt=".2f", 
                xticklabels=criteria, yticklabels=criteria)
    
    plt.title('Ma trận so sánh cặp giữa các tiêu chí', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded_img

def create_consistency_chart(cr):
    """Tạo biểu đồ thể hiện CR và so sánh với ngưỡng chấp nhận được (0.1)"""
    plt.figure(figsize=(8, 6))
    
    # Tạo dữ liệu
    categories = ['Chỉ số nhất quán (CR)', 'Ngưỡng chấp nhận được']
    values = [cr, 0.1]
    colors = ['green' if cr <= 0.1 else 'red', 'blue']
    
    # Vẽ biểu đồ
    bars = plt.bar(categories, values, color=colors)
    
    # Thêm nhãn giá trị
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
    plt.annotate('Ngưỡng 0.1', xy=(0, 0.1), xytext=(0, 0.12),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    # Thêm nhãn và tiêu đề
    plt.ylabel('Giá trị')
    plt.title('Chỉ số nhất quán (CR) so với ngưỡng chấp nhận được', fontsize=14, fontweight='bold')
    
    # Thêm chú thích
    if cr <= 0.1:
        plt.figtext(0.5, 0.01, 'Đánh giá: Nhất quán (CR ≤ 0.1)', ha='center', 
                   bbox={'facecolor':'green', 'alpha':0.3, 'pad':10}, fontsize=12)
    else:
        plt.figtext(0.5, 0.01, 'Đánh giá: Không nhất quán (CR > 0.1)', ha='center', 
                   bbox={'facecolor':'red', 'alpha':0.3, 'pad':10}, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded_img

@app.route('/input/<int:round_id>', methods=['POST'])
def input_candidates(round_id):
    num_candidates = int(request.form['num_candidates'])
    error_message = None

    # Lấy thông tin đợt tuyển dụng
    round_name = request.form.get('round_name')
    position = request.form.get('position')
    description = request.form.get('description')

    # Lấy thông tin ứng viên
    candidates_data = []
    try:
        for i in range(num_candidates):
            name = request.form[f'candidate_{i}']
            if not name.strip():  # Kiểm tra tên ứng viên không được để trống
                error_message = f"Tên ứng viên {i + 1} không được để trống."
                raise ValueError(error_message)

            values = []
            for j in range(num_criteria):
                value_str = request.form.get(f'candidate_{i}_criterion_{j}', '')
                if not value_str.strip():  # Kiểm tra trường điểm không được để trống
                    error_message = f"Điểm '{criteria[j]}' của ứng viên {i + 1} không được để trống."
                    raise ValueError(error_message)

                value = float(value_str)
                if value < 0:  # Điểm không được âm
                    error_message = f"Điểm '{criteria[j]}' của ứng viên {i + 1} không được nhỏ hơn 0."
                    raise ValueError(error_message)

                if value > optimal_values[j]:  # Kiểm tra điểm không vượt quá giá trị tối đa
                    error_message = f"Điểm '{criteria[j]}' của ứng viên {i + 1} vượt quá giá trị tối đa ({optimal_values[j]})."
                    raise ValueError(error_message)

                values.append(value)
            candidates_data.append({'name': name, 'values': values})

    except ValueError as e:
        # Nếu có lỗi, trả về trang index.html với thông báo lỗi và dữ liệu đã nhập
        return render_template('index.html', round_id=round_id, num_candidates=num_candidates, criteria=criteria,
                               pairwise=[[float(request.form.get(f'pairwise_{r}_{c}', 1.0)) for c in range(num_criteria)] for r in range(num_criteria)],
                               optimal_values=optimal_values, error_message=str(e),
                               round_name=round_name, position=position, description=description,
                               candidates_data=candidates_data)

    # Nếu không có lỗi, tiến hành lưu dữ liệu
    try:
        for candidate_data in candidates_data:
            name = candidate_data['name']
            values = candidate_data['values']
            candidate = Candidate(
                round_id=round_id,
                full_name=name,
                notes="N/A"
            )
            db.session.add(candidate)
            db.session.commit()
            candidate_data['id'] = candidate.candidate_id

    except Exception as e:
        db.session.rollback()
        return f"Lỗi khi lưu ứng viên: {str(e)}"

    # Lấy ma trận so sánh cặp từ form
    pairwise = np.zeros((num_criteria, num_criteria))
    try:
        for i in range(num_criteria):
            for j in range(num_criteria):
                pairwise[i][j] = float(request.form[f'pairwise_{i}_{j}'])
    except ValueError as e:
        return render_template('index.html', round_id=round_id, num_candidates=num_candidates, criteria=criteria,
                               pairwise=[[float(request.form.get(f'pairwise_{r}_{c}', 1.0)) for c in range(num_criteria)] for r in range(num_criteria)],
                               optimal_values=optimal_values, error_message="Vui lòng nhập đúng định dạng số cho ma trận so sánh.",
                               round_name=round_name, position=position, description=description,
                               candidates_data=candidates_data)

    # Tính toán AHP
    try:
        # 1. Tính trọng số tiêu chí
        col_sums = np.sum(pairwise, axis=0)
        normalized_matrix = pairwise / col_sums
        weights = np.mean(normalized_matrix, axis=1)

        # 2. Kiểm tra tính nhất quán
        lambda_max = np.sum(col_sums * weights)
        ci = (lambda_max - num_criteria) / (num_criteria - 1)
        # Assuming RI value for num_criteria is available. Replace 1.24 if num_criteria is different.
        # You might want to store RI values in a dictionary based on the number of criteria.
        ri_values = {
            3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45, 10: 1.49
        }
        ri = ri_values.get(num_criteria, 0.0) # Default to 0 if not found
        if num_criteria > 2 and ri != 0:
            cr = ci / ri
        else:
            cr = 0

        # Tạo các biểu đồ mới
        criterion_weights_image = create_criterion_weights_chart(criteria, weights)
        pairwise_matrix_image = create_pairwise_matrix_visualization(pairwise, criteria)
        consistency_chart_image = create_consistency_chart(cr)

        # Lưu ma trận so sánh tiêu chí
        matrix_data = json.dumps({
            "matrix": pairwise.tolist(),
            "weights": weights.tolist(),
        })
        criteria_matrix = CriteriaMatrix(
            round_id=round_id,
            matrix_data=matrix_data,
            consistency_ratio=cr
        )
        db.session.add(criteria_matrix)
        db.session.commit()

        # 3. Tính giá trị chuẩn hóa và xếp hạng
        banding = np.array([c['values'] for c in candidates_data]) / optimal_values
        banding_sums = np.sum(banding, axis=0)
        candidate_weights = banding / banding_sums
        scores = np.sum(candidate_weights * weights, axis=1)
        
        # Tạo biểu đồ kết quả đánh giá
        candidate_score_image = create_candidate_scores_chart(scores, [c['name'] for c in candidates_data])
        
        # Tạo biểu đồ dữ liệu thô
        raw_data_chart_image = create_raw_data_chart(
            [c['name'] for c in candidates_data],
            criteria,
            np.array([c['values'] for c in candidates_data])
        )

        # Tạo biểu đồ chi tiết tiêu chí cho từng ứng viên
        criteria_comparison_image = create_criteria_comparison_chart(
            [c['name'] for c in candidates_data],
            criteria,
            np.array([c['values'] for c in candidates_data]),
            weights
        )

        # 4. Xếp hạng và lưu vào bảng candidate_scores
        ranked = sorted(zip([c['id'] for c in candidates_data], [c['name'] for c in candidates_data], scores), key=lambda x: x[2], reverse=True)
        for rank, (candidate_id, name, score) in enumerate(ranked, 1):
            candidate_score = CandidateScore(
                round_id=round_id,
                candidate_id=candidate_id,
                total_score=score,
                ranking=rank
            )
            db.session.add(candidate_score)
        db.session.commit()

        if cr >= 0.1:
            # Nếu CR không đạt yêu cầu, không chuyển hướng và trả về thông báo lỗi cùng dữ liệu đã nhập
            return render_template('index.html',
                                   round_id=round_id,
                                   num_candidates=num_candidates,
                                   criteria=criteria,
                                   pairwise=pairwise.tolist(),  # Truyền lại ma trận để người dùng sửa
                                   optimal_values=optimal_values,
                                   error_message="Ma trận so sánh không phù hợp (CR >= 10%). Vui lòng nhập lại.",
                                   round_name=request.form.get('round_name'),
                                   position=request.form.get('position'),
                                   description=request.form.get('description'),
                                   candidates_data=candidates_data)
        else:
            # Trả kết quả nếu CR đạt yêu cầu
            ranked_display = [(name, score) for _, name, score in ranked]
            return render_template('results.html',
                                   round_id=round_id,
                                   weights=weights,
                                   criteria=criteria,
                                   cr=cr,
                                   ranked=ranked_display,
                                   criterion_weights_image=criterion_weights_image,
                                   candidate_score_image=candidate_score_image,
                                   criteria_comparison_image=criteria_comparison_image,
                                   raw_data_chart_image=raw_data_chart_image,
                                   pairwise_matrix_image=pairwise_matrix_image,
                                   consistency_chart_image=consistency_chart_image)

    except Exception as e:
        db.session.rollback()
        return f"Lỗi khi tính toán AHP: {str(e)}"

from fpdf import FPDF
import tempfile
import os
import json
import numpy as np
import matplotlib.pyplot as plt

@app.route('/export_report/<int:round_id>')
def export_report(round_id):
    """Xuất báo cáo kết quả tính toán dưới dạng PDF"""

    # Lấy thông tin đợt tuyển dụng
    round_info = RecruitmentRound.query.get_or_404(round_id)
    criteria_list = RecruitmentCriteria.query.filter_by(round_id=round_id).all()
    matrix = CriteriaMatrix.query.filter_by(round_id=round_id).first()
    scores = CandidateScore.query.filter_by(round_id=round_id).order_by(CandidateScore.ranking).all()

    # Xử lý ma trận
    weights = np.array([])
    if matrix and isinstance(matrix.matrix_data, str):
        try:
            matrix_data = json.loads(matrix.matrix_data)
            weights = np.array(matrix_data.get("weights", []))
        except:
            pass

    # Tạo biểu đồ trọng số tiêu chí
    criterion_weights_path = None
    if len(weights) > 0:
        plt.figure(figsize=(10, 6))
        plt.bar([c.criterion_name for c in criteria_list], weights, color='skyblue')
        plt.xlabel('Tiêu chí')
        plt.ylabel('Trọng số')
        plt.title('Trọng số các tiêu chí')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        fd, criterion_weights_path = tempfile.mkstemp(suffix='.png')
        plt.savefig(criterion_weights_path)
        os.close(fd)
        plt.close()

    # Tạo biểu đồ kết quả xếp hạng ứng viên
    candidate_scores_path = None
    if scores:
        ranked = []
        for score in scores:
            candidate = Candidate.query.get(score.candidate_id)
            ranked.append((candidate.full_name, score.total_score))

        plt.figure(figsize=(10, 6))
        plt.bar([name for name, _ in ranked], [score for _, score in ranked], color='lightgreen')
        plt.xlabel('Ứng viên')
        plt.ylabel('Điểm tổng')
        plt.title('Kết quả đánh giá ứng viên')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        fd, candidate_scores_path = tempfile.mkstemp(suffix='.png')
        plt.savefig(candidate_scores_path)
        os.close(fd)
        plt.close()

    # Tạo PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 16)
    pdf.cell(0, 10, f'Báo cáo đợt tuyển dụng: {round_info.round_name}', 0, 1, 'C')

    pdf.set_font('DejaVu', '', 12)
    pdf.cell(0, 10, f'Vị trí: {round_info.position}', 0, 1)
    pdf.cell(0, 10, f'Mô tả: {round_info.description}', 0, 1)
    pdf.cell(0, 10, f'Ngày tạo: {round_info.created_at.strftime("%d/%m/%Y")}', 0, 1)

    # Trọng số tiêu chí
    pdf.cell(0, 10, 'Trọng số các tiêu chí', 0, 1)
    for i, criterion in enumerate(criteria_list):
        if i < len(weights):
            pdf.cell(0, 8, f'{criterion.criterion_name}: {weights[i]:.4f}', 0, 1)

    # Chỉ số nhất quán
    if matrix:
        pdf.cell(0, 10, f'Chỉ số nhất quán CR: {matrix.consistency_ratio:.4f}', 0, 1)

    # Biểu đồ trọng số
    if criterion_weights_path:
        pdf.add_page()
        pdf.cell(0, 10, 'Biểu đồ trọng số tiêu chí', 0, 1, 'C')
        pdf.image(criterion_weights_path, x=10, y=40, w=190)
        try:
            os.remove(criterion_weights_path)
        except:
            pass

    pdf.add_page()
    pdf.cell(0, 10, 'Kết quả xếp hạng ứng viên', 0, 1)

    printed_ids = set()
    rank = 1

    for score in scores:
        if score.candidate_id not in printed_ids:
            candidate = Candidate.query.get(score.candidate_id)
            pdf.cell(0, 8, f'{rank}. {candidate.full_name}: {score.total_score:.4f}', 0, 1)
            printed_ids.add(score.candidate_id)
            rank += 1


    # Biểu đồ kết quả xếp hạng
    if candidate_scores_path:
        pdf.add_page()
        pdf.cell(0, 10, 'Biểu đồ kết quả đánh giá ứng viên', 0, 1, 'C')
        pdf.image(candidate_scores_path, x=10, y=40, w=190)
        try:
            os.remove(candidate_scores_path)
        except:
            pass

    # Xuất file PDF tạm và gửi về cho người dùng
    fd, temp_pdf_path = tempfile.mkstemp(suffix='.pdf')
    os.close(fd)
    pdf.output(temp_pdf_path)

    return send_file(
        temp_pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'bao-cao-tuyen-dung-{round_info.round_name}.pdf',
        max_age=0
    )

# Xem lịch sử đợt tuyển dụng
@app.route('/history')
def history():
    rounds = RecruitmentRound.query.all()
    return render_template('history.html', rounds=rounds)

# Xem chi tiết đợt tuyển dụng
@app.route('/round/<int:round_id>')
def round_detail(round_id):
    round = RecruitmentRound.query.get_or_404(round_id)
    criteria_list = RecruitmentCriteria.query.filter_by(round_id=round_id).all()
    matrix = CriteriaMatrix.query.filter_by(round_id=round_id).first()
    candidates = Candidate.query.filter_by(round_id=round_id).all()
    scores = CandidateScore.query.filter_by(round_id=round_id).order_by(CandidateScore.ranking).all()

    # Xử lý matrix_data (vì cột là db.JSON, dữ liệu trả về là chuỗi)
    if matrix and isinstance(matrix.matrix_data, str):
        try:
            matrix.matrix_data = json.loads(matrix.matrix_data)
        except json.JSONDecodeError:
            matrix.matrix_data = {"matrix": []}  # Giá trị mặc định nếu không phân tích được
    elif not matrix:
        matrix = CriteriaMatrix(matrix_data={"matrix": []}, consistency_ratio=0.0)  # Giá trị mặc định nếu matrix không tồn tại

    # Lấy tên ứng viên từ candidate_id trong scores
    ranked = []
    for score in scores:
        candidate = Candidate.query.get(score.candidate_id)
        ranked.append((candidate.full_name, score.total_score))

    return render_template('round_detail.html', round=round, criteria=criteria_list, matrix=matrix, candidates=candidates, ranked=ranked)

# Xóa đợt tuyển dụng
@app.route('/delete_round/<int:round_id>')
def delete_round(round_id):
    try:
        # Xóa dữ liệu liên quan trước
        CandidateScore.query.filter_by(round_id=round_id).delete()
        Candidate.query.filter_by(round_id=round_id).delete()
        CriteriaMatrix.query.filter_by(round_id=round_id).delete()
        RecruitmentCriteria.query.filter_by(round_id=round_id).delete()
        RecruitmentRound.query.filter_by(round_id=round_id).delete()
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return f"Lỗi khi xóa đợt tuyển dụng: {str(e)}"

    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True)
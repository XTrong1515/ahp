from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
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
# from flask import session # Không cần import lại nếu đã có ở dòng đầu

app = Flask(__name__)
app.secret_key = '105008truonganhminhtrong' # Thay bằng chuỗi bí mật thực sự

# Cấu hình kết nối MySQL
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Theanh412%40@localhost:3306/recruitment_ahp'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost:3306/recruitment_ahp'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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
    # is_custom = db.Column(db.Boolean, default=False) # Cột này có thể không còn cần thiết
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class CriteriaMatrix(db.Model):
    __tablename__ = 'criteria_matrix'
    matrix_id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey('recruitment_rounds.round_id'), nullable=False)
    matrix_data = db.Column(db.JSON, nullable=False)
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

# --- CÁC HÀM HELPER VÀ GIÁ TRỊ RI ---
ri_values = {
    1: 0.00, 2: 0.00, 3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25,
    7: 1.35, 8: 1.40, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54,
    13: 1.56, 14: 1.57, 15: 1.59
}

def calculate_priority_vector(matrix):
    matrix = np.array(matrix, dtype=float) # Đảm bảo kiểu float
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        return np.array([])
    
    col_sums = np.sum(matrix, axis=0)
    if np.any(col_sums == 0): # Tránh chia cho 0
        # Trong AHP, điều này không nên xảy ra nếu ma trận được nhập đúng
        # Có thể trả về trọng số bằng nhau hoặc báo lỗi
        # For now, if a column sum is zero, it implies an issue with matrix construction
        # or that the matrix is not a valid Saaty matrix.
        # Returning equal weights or raising an error might be options.
        # For robustness, let's assume if this happens, priority vector calculation is problematic.
        return np.full(matrix.shape[0], 1/matrix.shape[0]) # Default to equal weights as a fallback

    normalized_matrix = matrix / col_sums
    weights = np.mean(normalized_matrix, axis=1)
    return weights

def calculate_consistency_ratio(matrix):
    matrix = np.array(matrix, dtype=float) # Đảm bảo kiểu float
    n = matrix.shape[0]

    if n == 0:
        return 0.0 
    if n <= 2: # Ma trận 1x1 hoặc 2x2 luôn nhất quán
        return 0.0

    weights = calculate_priority_vector(matrix)
    if weights.size == 0 or weights.shape[0] != n: # Kiểm tra weights hợp lệ
        return float('inf') 

    # Tính lambda_max
    # Cách 1: weighted_sum_vector = matrix @ weights 
    #          lambda_max = np.mean(weighted_sum_vector / weights) # Có thể không ổn định nếu weights có phần tử 0
    # Cách 2: (Thường dùng trong AHP)
    # Eigenvector method: Ax = lambda_max * x
    # lambda_max = sum of (each element of (matrix @ weights) / corresponding element of weights) / n
    # Or, more directly related to consistency:
    aw = np.dot(matrix, weights)
    # Ensure no division by zero if any weight is zero (should not happen with valid priority vector)
    if np.any(weights == 0):
        lambda_max = n # Fallback, implies perfect consistency or error in weights
    else:
        lambda_max = np.mean(aw / weights)


    if n - 1 == 0: # Should be caught by n <= 2, but as a safeguard
        return 0.0 if lambda_max == n else float('inf')
        
    ci = (lambda_max - n) / (n - 1)
    
    ri = ri_values.get(n)
    if ri is None: 
        ri = ri_values.get(15, 1.59) # Mặc định cho n > 15
    
    if ri == 0: # Chỉ xảy ra nếu n <= 2 và logic ở trên chưa bắt được, hoặc ri_values bị sửa đổi
        return 0.0 if ci == 0 else float('inf') # Nếu CI cũng là 0 thì nhất quán, ngược lại là lỗi
             
    cr = ci / ri
    return cr

# --- CÁC BIẾN TOÀN CỤC CŨ (Đã comment out hoặc sẽ được thay thế) ---
# criteria = ["Kiến thức chuyên môn", ...] 
# num_criteria_global = len(criteria) # (Sử dụng tên khác để tránh nhầm lẫn với num_criteria trong session)
# default_pairwise = np.array([...])
# optimal_values = np.array([100, 10, 10, 10, 5, 10]) # Biến này có thể không còn dùng trong quy trình AHP mới

# Kiểm tra kết nối đến database
@app.route('/check_connection')
def check_connection():
    try:
        RecruitmentRound.query.first()
        return "Database connection is successful."
    except Exception as e:
        return f"Database connection failed: {str(e)}"

# Trang chính: Tạo đợt tuyển dụng
@app.route('/', methods=['GET', 'POST'])
def index_or_create_round():
    if request.method == 'POST':
        round_name = request.form.get('round_name','').strip()
        description = request.form.get('description', '').strip()
        position = request.form.get('position','').strip()

        if not round_name or not position:
            flash("Tên đợt tuyển dụng và vị trí không được để trống.", "danger")
            return render_template('index.html', round_name=round_name, description=description, position=position)

        try:
            new_round = RecruitmentRound(
                round_name=round_name,
                description=description,
                position=position
            )
            db.session.add(new_round)
            db.session.commit()
            
            session.clear() # Xóa session cũ khi bắt đầu đợt mới
            session['round_id'] = new_round.round_id
            session['round_name'] = new_round.round_name
            flash(f"Đợt tuyển dụng '{round_name}' đã được tạo thành công!", "success")
            return redirect(url_for('setup_criteria_count'))
        except Exception as e:
            db.session.rollback()
            flash(f"Lỗi khi tạo đợt tuyển dụng: {str(e)}", "danger")
            return render_template('index.html', round_name=round_name, description=description, position=position)
    
    # Xóa session khi truy cập trang chủ bằng GET để bắt đầu mới (tuỳ chọn)
    # session.pop('round_id', None) 
    # session.pop('round_name', None)
    # ... xóa các key khác của AHP process
    return render_template('index.html')

@app.route('/setup_criteria_count', methods=['GET', 'POST'])
def setup_criteria_count():
    if 'round_id' not in session:
        flash("Vui lòng tạo hoặc chọn một đợt tuyển dụng trước.", "warning")
        return redirect(url_for('index_or_create_round'))

    if request.method == 'POST':
        try:
            num_criteria = int(request.form['num_criteria'])
            if num_criteria <= 1:
                flash("Số lượng tiêu chí phải lớn hơn 1.", "warning")
                return render_template('setup_criteria_count.html', 
                                       round_name=session.get('round_name'),
                                       num_criteria_value=request.form.get('num_criteria'))
            session['num_criteria'] = num_criteria
            return redirect(url_for('setup_criteria_details'))
        except ValueError:
            flash("Vui lòng nhập một số hợp lệ cho số lượng tiêu chí.", "danger")
            return render_template('setup_criteria_count.html', 
                                   round_name=session.get('round_name'))
                                   
    return render_template('setup_criteria_count.html', round_name=session.get('round_name'))

@app.route('/setup_criteria_details', methods=['GET', 'POST'])
def setup_criteria_details():
    if 'round_id' not in session or 'num_criteria' not in session:
        flash("Thông tin đợt tuyển dụng hoặc số lượng tiêu chí bị thiếu. Vui lòng thử lại.", "warning")
        return redirect(url_for('index_or_create_round'))

    num_criteria = session.get('num_criteria', 0)
    if num_criteria == 0: # Đảm bảo num_criteria hợp lệ
        return redirect(url_for('setup_criteria_count'))

    # Lấy giá trị đã nhập nếu có (khi render lại do lỗi)
    criteria_names_input = [''] * num_criteria
    matrix_values_input = np.ones((num_criteria, num_criteria)).tolist()


    if request.method == 'POST':
        criteria_names = [request.form.get(f'criterion_name_{i}', '').strip() for i in range(num_criteria)]
        if any(not name for name in criteria_names):
            flash("Tên tiêu chí không được để trống.", "danger")
            # Truyền lại giá trị đã nhập để người dùng không phải nhập lại tất cả
            for i in range(num_criteria): matrix_values_input[i] = [request.form.get(f'criteria_pairwise_{i}_{j}', 1.0) for j in range(num_criteria)]
            return render_template('setup_criteria_details.html', 
                                   num_criteria=num_criteria, 
                                   round_name=session.get('round_name'),
                                   criteria_names_input=criteria_names, # tên đã nhập
                                   matrix_values_input=matrix_values_input # ma trận đã nhập
                                   )

        pairwise_matrix_criteria_list = []
        try:
            for i in range(num_criteria):
                row = []
                for j in range(num_criteria):
                    value_str = request.form.get(f'criteria_pairwise_{i}_{j}', '1.0') # Mặc định là 1 nếu thiếu
                    val = float(value_str)
                    if val <= 0: # Giá trị phải dương
                        flash(f"Giá trị so sánh giữa '{criteria_names[i]}' và '{criteria_names[j]}' phải là số dương.", "danger")
                        for r_idx in range(num_criteria): matrix_values_input[r_idx] = [request.form.get(f'criteria_pairwise_{r_idx}_{c_idx}', 1.0) for c_idx in range(num_criteria)]
                        return render_template('setup_criteria_details.html', num_criteria=num_criteria, criteria_names_input=criteria_names, matrix_values_input=matrix_values_input, round_name=session.get('round_name'))
                    row.append(val)
                pairwise_matrix_criteria_list.append(row)
        except ValueError:
            flash("Giá trị trong ma trận không hợp lệ. Vui lòng nhập số.", "danger")
            # Truyền lại giá trị đã nhập
            for i in range(num_criteria): matrix_values_input[i] = [request.form.get(f'criteria_pairwise_{i}_{j}', 1.0) for j in range(num_criteria)]
            return render_template('setup_criteria_details.html', num_criteria=num_criteria, criteria_names_input=criteria_names, matrix_values_input=matrix_values_input, round_name=session.get('round_name'))

        pairwise_matrix_criteria = np.array(pairwise_matrix_criteria_list)
        
        cr_criteria = calculate_consistency_ratio(pairwise_matrix_criteria)
        weights_criteria = calculate_priority_vector(pairwise_matrix_criteria)

        session['criteria_names'] = criteria_names
        session['pairwise_matrix_criteria'] = pairwise_matrix_criteria.tolist()
        session['weights_criteria'] = weights_criteria.tolist()
        session['cr_criteria'] = cr_criteria
        
        round_id = session['round_id']
        # Xóa tiêu chí cũ của round này trước khi thêm mới (nếu user quay lại và sửa)
        RecruitmentCriteria.query.filter_by(round_id=round_id).delete()
        for name in criteria_names:
            criterion_db = RecruitmentCriteria(round_id=round_id, criterion_name=name)
            db.session.add(criterion_db)
        
        existing_matrix = CriteriaMatrix.query.filter_by(round_id=round_id).first()
        matrix_data_json = json.dumps({
            "matrix": pairwise_matrix_criteria.tolist(),
            "weights": weights_criteria.tolist(),
            "criteria_names_at_creation": criteria_names
        })
        if existing_matrix:
            existing_matrix.matrix_data = matrix_data_json
            existing_matrix.consistency_ratio = cr_criteria
        else:
            new_criteria_matrix_db = CriteriaMatrix(
                round_id=round_id,
                matrix_data=matrix_data_json,
                consistency_ratio=cr_criteria
            )
            db.session.add(new_criteria_matrix_db)
        db.session.commit()

        if cr_criteria < 0.1:
            flash(f"Ma trận tiêu chí nhất quán (CR = {cr_criteria:.4f}).", "success")
            return redirect(url_for('setup_candidates_count'))
        else:
            flash(f"CR của ma trận tiêu chí ({cr_criteria:.4f}) >= 0.1. Vui lòng kiểm tra lại.", "warning")
            return render_template('setup_criteria_details.html', 
                                   num_criteria=num_criteria, 
                                   criteria_names_input=criteria_names, 
                                   matrix_values_input=pairwise_matrix_criteria.tolist(), 
                                   round_name=session.get('round_name'))
    
    # Cho GET request
    # Nếu có giá trị cũ trong session (ví dụ user back), hiển thị lại
    if 'criteria_names' in session and len(session['criteria_names']) == num_criteria :
        criteria_names_input = session['criteria_names']
    if 'pairwise_matrix_criteria' in session and len(session['pairwise_matrix_criteria']) == num_criteria:
         matrix_values_input = session['pairwise_matrix_criteria']


    return render_template('setup_criteria_details.html', 
                           num_criteria=num_criteria, 
                           criteria_names_input=criteria_names_input,
                           matrix_values_input=matrix_values_input,
                           round_name=session.get('round_name'))

@app.route('/setup_candidates_count', methods=['GET', 'POST'])
def setup_candidates_count():
    if 'round_id' not in session:
        return redirect(url_for('index_or_create_round'))
    if session.get('cr_criteria', 1.0) >= 0.1:
        flash("Ma trận tiêu chí chưa nhất quán. Vui lòng sửa trước khi tiếp tục.", "warning")
        return redirect(url_for('setup_criteria_details'))

    if request.method == 'POST':
        try:
            num_candidates = int(request.form['num_candidates'])
            if num_candidates <= 1:
                flash("Số lượng ứng viên phải lớn hơn 1.", "warning")
                return render_template('setup_candidates_count.html', 
                                       round_name=session.get('round_name'),
                                       criteria_names=session.get('criteria_names'),
                                       num_candidates_value = request.form.get('num_candidates'))
            session['num_candidates'] = num_candidates
            return redirect(url_for('setup_candidate_names'))
        except ValueError:
            flash("Vui lòng nhập một số hợp lệ cho số lượng ứng viên.", "danger")
            return render_template('setup_candidates_count.html', 
                                   round_name=session.get('round_name'),
                                   criteria_names=session.get('criteria_names'))

    return render_template('setup_candidates_count.html', 
                           round_name=session.get('round_name'),
                           criteria_names=session.get('criteria_names'))

@app.route('/setup_candidate_names', methods=['GET', 'POST'])
def setup_candidate_names():
    if 'round_id' not in session or 'num_candidates' not in session:
        return redirect(url_for('setup_candidates_count'))
    
    num_candidates = session.get('num_candidates', 0)
    if num_candidates == 0:
        return redirect(url_for('setup_candidates_count'))

    candidate_names_input = [''] * num_candidates

    if request.method == 'POST':
        candidate_names = [request.form.get(f'candidate_name_{i}', '').strip() for i in range(num_candidates)]
        if any(not name for name in candidate_names):
            flash("Tên ứng viên không được để trống.", "danger")
            return render_template('setup_candidate_names.html', 
                                   num_candidates=num_candidates, 
                                   round_name=session.get('round_name'),
                                   candidate_names_input=candidate_names)

        session['candidate_names'] = candidate_names
        session['candidate_pairwise_matrices_details'] = [None] * len(session.get('criteria_names', []))
        
        round_id = session['round_id']
        # Xóa ứng viên cũ của round này trước khi thêm mới
        Candidate.query.filter_by(round_id=round_id).delete() # Cẩn thận với việc xóa dữ liệu nếu có liên kết
        db.session.commit() # Commit việc xóa trước khi thêm

        candidate_ids = []
        for name in candidate_names:
            candidate_db = Candidate(round_id=round_id, full_name=name)
            db.session.add(candidate_db)
            db.session.flush() 
            candidate_ids.append(candidate_db.candidate_id)
        db.session.commit()
        session['candidate_ids'] = candidate_ids

        return redirect(url_for('input_candidate_comparison_for_criterion', criterion_idx=0))
    
    # Cho GET request
    if 'candidate_names' in session and len(session['candidate_names']) == num_candidates:
        candidate_names_input = session['candidate_names']
       
    return render_template('setup_candidate_names.html', 
                           num_candidates=num_candidates, 
                           round_name=session.get('round_name'),
                           candidate_names_input=candidate_names_input)


@app.route('/input_candidate_comparison_for_criterion/<int:criterion_idx>', methods=['GET', 'POST'])
def input_candidate_comparison_for_criterion(criterion_idx):
    required_sessions = ['round_id', 'criteria_names', 'candidate_names', 'candidate_pairwise_matrices_details']
    for key in required_sessions:
        if key not in session:
            flash(f"Thiếu thông tin cần thiết ({key}). Vui lòng bắt đầu lại.", "warning")
            return redirect(url_for('index_or_create_round'))

    criteria_names = session.get('criteria_names', [])
    candidate_names = session.get('candidate_names', [])
    num_candidates = len(candidate_names)
    candidate_matrices_details = session.get('candidate_pairwise_matrices_details', [])

    # Đảm bảo candidate_matrices_details có đủ độ dài
    if len(candidate_matrices_details) != len(criteria_names):
        candidate_matrices_details = [None] * len(criteria_names)
        session['candidate_pairwise_matrices_details'] = candidate_matrices_details


    if not (0 <= criterion_idx < len(criteria_names)):
        # Kiểm tra nếu đã hoàn thành tất cả các tiêu chí
        all_done_and_consistent = True
        if len(candidate_matrices_details) != len(criteria_names): # Chưa đủ số ma trận
            all_done_and_consistent = False
        else:
            for i, detail in enumerate(candidate_matrices_details):
                if detail is None: # Một tiêu chí chưa có ma trận
                     # flash(f"Vui lòng hoàn thành ma trận so sánh cho tiêu chí '{criteria_names[i]}'.", "info")
                     return redirect(url_for('input_candidate_comparison_for_criterion', criterion_idx=i))
                if detail['cr'] >= 0.1:
                    all_done_and_consistent = False # Vẫn cho qua nhưng sẽ báo lỗi ở trang kết quả

        if all_done_and_consistent or (all(d is not None for d in candidate_matrices_details)): # Nếu đã điền hết (dù có CR lỗi)
            return redirect(url_for('calculate_final_ranking'))
        else: # Trường hợp lạ, redirect về trang đầu
            flash("Có lỗi trong quy trình, vui lòng bắt đầu lại.", "danger")
            return redirect(url_for('index_or_create_round'))


    current_criterion_name = criteria_names[criterion_idx]
    matrix_values_input = np.ones((num_candidates, num_candidates)).tolist() # Default

    if request.method == 'POST':
        pairwise_matrix_candidate_list = []
        try:
            for i in range(num_candidates):
                row = []
                for j in range(num_candidates):
                    value_str = request.form.get(f'candidate_pairwise_{i}_{j}', '1.0')
                    val = float(value_str)
                    if val <= 0:
                        flash(f"Giá trị so sánh giữa '{candidate_names[i]}' và '{candidate_names[j]}' phải là số dương.", "danger")
                        # Truyền lại giá trị đã nhập
                        for r_idx in range(num_candidates): matrix_values_input[r_idx] = [request.form.get(f'candidate_pairwise_{r_idx}_{c_idx}', 1.0) for c_idx in range(num_candidates)]
                        return render_template('input_candidate_comparison.html', criterion_name=current_criterion_name, criterion_idx=criterion_idx, candidate_names=candidate_names, matrix_values_input=matrix_values_input, round_name=session.get('round_name'))
                    row.append(val)
                pairwise_matrix_candidate_list.append(row)
        except ValueError:
            flash("Giá trị trong ma trận không hợp lệ. Vui lòng nhập số.", "danger")
            # Truyền lại giá trị đã nhập
            for i in range(num_candidates): matrix_values_input[i] = [request.form.get(f'candidate_pairwise_{i}_{j}', 1.0) for j in range(num_candidates)]
            return render_template('input_candidate_comparison.html', criterion_name=current_criterion_name, criterion_idx=criterion_idx, candidate_names=candidate_names, matrix_values_input=matrix_values_input, round_name=session.get('round_name'))

        pairwise_matrix_candidate = np.array(pairwise_matrix_candidate_list)
        cr_candidate = calculate_consistency_ratio(pairwise_matrix_candidate)
        weights_candidate_local = calculate_priority_vector(pairwise_matrix_candidate)

        candidate_matrices_details[criterion_idx] = {
            'matrix': pairwise_matrix_candidate.tolist(),
            'cr': cr_candidate,
            'weights': weights_candidate_local.tolist(),
            'criterion_name': current_criterion_name
        }
        session['candidate_pairwise_matrices_details'] = candidate_matrices_details

        if cr_candidate >= 0.1:
            flash(f"CR cho ứng viên theo tiêu chí '{current_criterion_name}' ({cr_candidate:.4f}) >= 0.1. Vui lòng kiểm tra lại.", "warning")
            return render_template('input_candidate_comparison.html',
                                   criterion_name=current_criterion_name,
                                   criterion_idx=criterion_idx,
                                   candidate_names=candidate_names,
                                   matrix_values_input=pairwise_matrix_candidate.tolist(),
                                   round_name=session.get('round_name'))
        
        flash(f"Ma trận cho tiêu chí '{current_criterion_name}' đã được lưu (CR = {cr_candidate:.4f}).", "success")
        next_criterion_idx = criterion_idx + 1
        if next_criterion_idx < len(criteria_names):
            return redirect(url_for('input_candidate_comparison_for_criterion', criterion_idx=next_criterion_idx))
        else:
            return redirect(url_for('calculate_final_ranking'))

    # Cho GET request: lấy lại giá trị đã nhập nếu có
    if candidate_matrices_details[criterion_idx] is not None:
        matrix_values_input = candidate_matrices_details[criterion_idx]['matrix']

    return render_template('input_candidate_comparison.html', 
                           criterion_name=current_criterion_name,
                           criterion_idx=criterion_idx,
                           candidate_names=candidate_names,
                           matrix_values_input=matrix_values_input,
                           round_name=session.get('round_name'))

@app.route('/calculate_final_ranking')
def calculate_final_ranking():
    required_sessions = ['round_id', 'criteria_names', 'weights_criteria', 'cr_criteria', 
                         'candidate_names', 'candidate_ids', 'candidate_pairwise_matrices_details']
    for key in required_sessions:
        if key not in session:
            flash(f"Thiếu dữ liệu phiên làm việc: {key}. Vui lòng bắt đầu lại.", "danger")
            return redirect(url_for('index_or_create_round'))

    cr_criteria = session.get('cr_criteria', 1.0) # Mặc định là không nhất quán
    candidate_matrices_details = session.get('candidate_pairwise_matrices_details', [])
    
    all_consistent = cr_criteria < 0.1
    inconsistent_candidate_matrices_info = []
    
    criteria_names = session.get('criteria_names', [])
    if not candidate_matrices_details or len(candidate_matrices_details) != len(criteria_names):
        first_missing_idx = 0
        if candidate_matrices_details: # Nếu list không rỗng
            first_missing_idx = next((i for i, detail in enumerate(candidate_matrices_details) if detail is None), len(candidate_matrices_details))
        
        if first_missing_idx < len(criteria_names):
            flash(f"Chưa hoàn thành việc nhập liệu cho ma trận ứng viên của tiêu chí '{criteria_names[first_missing_idx]}'.", "warning")
            return redirect(url_for('input_candidate_comparison_for_criterion', criterion_idx=first_missing_idx))
        else: # Trường hợp lạ
            flash("Dữ liệu ma trận ứng viên không đầy đủ. Vui lòng thử lại.", "danger")
            return redirect(url_for('index_or_create_round'))

    for idx, detail in enumerate(candidate_matrices_details):
        # detail đã được kiểm tra not None ở trên
        if detail['cr'] >= 0.1:
            all_consistent = False
            inconsistent_candidate_matrices_info.append(f"Tiêu chí '{detail.get('criterion_name', criteria_names[idx])}': CR = {detail['cr']:.4f}")
            
    weights_criteria = np.array(session.get('weights_criteria', []))
    candidate_names = session.get('candidate_names', [])
    candidate_ids = session.get('candidate_ids', [])
    
    num_candidates = len(candidate_names)
    num_criteria = len(criteria_names)

    if weights_criteria.shape[0] != num_criteria:
        flash("Lỗi dữ liệu trọng số tiêu chí.", "danger")
        return redirect(url_for('setup_criteria_details'))

    candidate_performance_matrix = np.zeros((num_candidates, num_criteria))
    for crit_idx in range(num_criteria):
        detail = candidate_matrices_details[crit_idx]
        local_weights = np.array(detail['weights'])
        if local_weights.shape[0] == num_candidates:
            candidate_performance_matrix[:, crit_idx] = local_weights
        else: # Lỗi, gán trọng số bằng nhau
            flash(f"Lỗi kích thước trọng số ứng viên cho tiêu chí '{criteria_names[crit_idx]}'. Đã đặt giá trị mặc định.", "warning")
            candidate_performance_matrix[:, crit_idx] = np.ones(num_candidates) / num_candidates

    final_scores = candidate_performance_matrix @ weights_criteria
    
    ranked_candidates_list = []
    for i in range(num_candidates):
        ranked_candidates_list.append({
            'id': candidate_ids[i],
            'name': candidate_names[i],
            'score': final_scores[i]
        })
    
    ranked_candidates_display = sorted(ranked_candidates_list, key=lambda x: x['score'], reverse=True)

    round_id = session['round_id']
    if all_consistent:
        CandidateScore.query.filter_by(round_id=round_id).delete()
        for rank_idx, cand_data in enumerate(ranked_candidates_display):
            candidate_score_db = CandidateScore(
                round_id=round_id,
                candidate_id=cand_data['id'],
                total_score=cand_data['score'],
                ranking=rank_idx + 1
            )
            db.session.add(candidate_score_db)
        db.session.commit()
        flash("Kết quả đã được tính toán và lưu trữ thành công!", "success")
    else:
        flash("Một số ma trận không nhất quán. Kết quả chỉ mang tính tham khảo và chưa được lưu vào lịch sử chính thức.", "info")


    # Tạo các biểu đồ
    criterion_weights_image = None
    if weights_criteria.size > 0 and len(criteria_names) == weights_criteria.size:
         criterion_weights_image = create_criterion_weights_chart(criteria_names, weights_criteria)
    
    candidate_score_image = None
    cand_scores_for_chart = [c['score'] for c in ranked_candidates_display]
    cand_names_for_chart = [c['name'] for c in ranked_candidates_display]
    if cand_scores_for_chart and cand_names_for_chart:
        candidate_score_image = create_candidate_scores_chart(cand_scores_for_chart, cand_names_for_chart)
    
    # Các biểu đồ khác bạn có thể tạo tương tự
    # pairwise_matrix_image = create_pairwise_matrix_visualization(np.array(session.get('pairwise_matrix_criteria', [])), criteria_names)
    # consistency_chart_image = create_consistency_chart(cr_criteria)
    
    # session.pop('ahp_data_active', None) # Ví dụ xóa cờ đang trong quy trình AHP
    
    return render_template('final_results.html',
                           round_name=session.get('round_name'),
                           ranked_candidates=ranked_candidates_display,
                           criteria_names=criteria_names,
                           weights_criteria=weights_criteria.tolist(),
                           cr_criteria=cr_criteria,
                           candidate_names=candidate_names,
                           candidate_matrices_details=candidate_matrices_details,
                           all_consistent=all_consistent,
                           inconsistent_candidate_matrices_info=inconsistent_candidate_matrices_info,
                           pairwise_matrix_criteria=session.get('pairwise_matrix_criteria', []),
                           criterion_weights_image=criterion_weights_image,
                           candidate_score_image=candidate_score_image
                           # Thêm các ảnh khác nếu có
                           )

# --- HÀM TẠO BIỂU ĐỒ (GIỮ NGUYÊN HOẶC CHỈNH SỬA NHẸ NẾU CẦN) ---
def create_criterion_weights_chart(criteria, weights):
    if not criteria or not isinstance(weights, (list, np.ndarray)) or not len(weights) or len(criteria) != len(weights):
        return None
    plt.figure(figsize=(10, 6))
    bars = plt.bar(criteria, weights, color='skyblue')
    plt.xlabel('Tiêu chí', fontsize=12)
    plt.ylabel('Trọng số', fontsize=12)
    plt.title('Trọng số các tiêu chí', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    if len(weights) > 0 and max(weights) > 0 : plt.ylim(0, max(weights) * 1.2)
    else: plt.ylim(0,0.1)


    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded_img

def create_candidate_scores_chart(scores, candidate_names):
    if not candidate_names or not isinstance(scores, (list, np.ndarray)) or not len(scores) or len(candidate_names) != len(scores):
        return None
    plt.figure(figsize=(10, 6))
    bars = plt.bar(candidate_names, scores, color='lightcoral')
    plt.xlabel('Ứng viên', fontsize=12)
    plt.ylabel('Tổng điểm', fontsize=12)
    plt.title('Điểm số tổng của các ứng viên', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    if len(scores) > 0 and max(scores) > 0: plt.ylim(0, max(scores) * 1.2)
    else: plt.ylim(0,0.1)


    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded_img

# Các hàm create_criteria_comparison_chart, create_raw_data_chart, 
# create_pairwise_matrix_visualization, create_consistency_chart
# bạn có thể giữ lại và gọi chúng trong `calculate_final_ranking` nếu muốn hiển thị các biểu đồ đó.
# Hãy đảm bảo chúng nhận đúng tham số từ dữ liệu AHP mới.

# --- ROUTE CŨ CẦN XEM XÉT KỸ ---
# @app.route('/input/<int:round_id>', methods=['POST'])
# def input_candidates(round_id):
#     # Route này có thể không còn cần thiết nữa nếu bạn hoàn toàn chuyển sang quy trình mới.
#     # Nếu vẫn muốn giữ lại, cần tích hợp rất cẩn thận.
#     pass


# --- CÁC ROUTE QUẢN LÝ LỊCH SỬ, XUẤT PDF ---
# (Cần cập nhật để tương thích với dữ liệu mới, ví dụ: các tiêu chí động,
# ma trận so sánh ứng viên được lưu trong session hoặc DB nếu bạn mở rộng)

from fpdf import FPDF # Đảm bảo import FPDF
import tempfile
import os

@app.route('/export_report/<int:round_id>')
def export_report(round_id):
    round_info = RecruitmentRound.query.get_or_404(round_id)
    criteria_list_db = RecruitmentCriteria.query.filter_by(round_id=round_id).all() # Lấy từ DB
    criteria_matrix_db = CriteriaMatrix.query.filter_by(round_id=round_id).first()
    candidate_scores_db = CandidateScore.query.filter_by(round_id=round_id).order_by(CandidateScore.ranking).all()

    pdf = FPDF()
    pdf.add_page()
    try: # Thêm font hỗ trợ Unicode
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.set_font('DejaVu', '', 16)
    except RuntimeError: # Fallback nếu font không tìm thấy
        pdf.set_font('Arial', 'B', 16)
        flash("Font DejaVu Sans không tìm thấy, báo cáo PDF có thể không hiển thị đúng tiếng Việt.", "warning")


    pdf.cell(0, 10, f'Báo Cáo Tuyển Dụng: {round_info.round_name}', 0, 1, 'C')
    pdf.set_font('DejaVu' if 'DejaVu' in pdf.font_family else 'Arial', '', 12)
    pdf.cell(0, 10, f'Vị trí: {round_info.position}', 0, 1)
    if round_info.description:
        pdf.multi_cell(0, 10, f'Mô tả: {round_info.description}') # Dùng multi_cell cho text dài
    pdf.cell(0, 10, f'Ngày tạo: {round_info.created_at.strftime("%d/%m/%Y %H:%M:%S")}', 0, 1)
    pdf.ln(5)

    # Trọng số và CR của Tiêu chí
    if criteria_matrix_db and criteria_matrix_db.matrix_data:
        matrix_data = criteria_matrix_db.matrix_data # Đây là dict từ JSON
        criteria_names_from_matrix = matrix_data.get("criteria_names_at_creation", [c.criterion_name for c in criteria_list_db])
        weights_criteria = matrix_data.get("weights", [])
        
        pdf.set_font('DejaVu' if 'DejaVu' in pdf.font_family else 'Arial', 'B', 12)
        pdf.cell(0, 10, '1. Phân Tích Tiêu Chí', 0, 1)
        pdf.set_font('DejaVu' if 'DejaVu' in pdf.font_family else 'Arial', '', 11)
        pdf.cell(0, 8, f'Chỉ số nhất quán (CR) ma trận tiêu chí: {criteria_matrix_db.consistency_ratio:.4f}', 0, 1)
        
        pdf.cell(0, 8, 'Trọng số các tiêu chí:', 0, 1)
        for i, name in enumerate(criteria_names_from_matrix):
            if i < len(weights_criteria):
                pdf.cell(0, 7, f'  - {name}: {weights_criteria[i]:.4f}', 0, 1)
        pdf.ln(5)
    
    # Kết quả Xếp Hạng Ứng Viên
    if candidate_scores_db:
        pdf.set_font('DejaVu' if 'DejaVu' in pdf.font_family else 'Arial', 'B', 12)
        pdf.cell(0, 10, '2. Kết Quả Xếp Hạng Ứng Viên', 0, 1)
        pdf.set_font('DejaVu' if 'DejaVu' in pdf.font_family else 'Arial', '', 11)
        
        # Header bảng
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(15, 7, 'Hạng', 1, 0, 'C', 1)
        pdf.cell(100, 7, 'Tên Ứng Viên', 1, 0, 'C', 1)
        pdf.cell(65, 7, 'Điểm Tổng Hợp AHP', 1, 1, 'C', 1)

        for score_entry in candidate_scores_db:
            candidate_info = Candidate.query.get(score_entry.candidate_id)
            pdf.cell(15, 7, str(score_entry.ranking), 1, 0, 'C')
            pdf.cell(100, 7, candidate_info.full_name if candidate_info else 'N/A', 1, 0, 'L')
            pdf.cell(65, 7, f'{score_entry.total_score:.4f}', 1, 1, 'R')
        pdf.ln(5)

    # (Tùy chọn) Thêm chi tiết ma trận so sánh ứng viên nếu bạn có lưu chúng vào DB
    # Hoặc nếu chúng đang có trong session khi export được gọi từ trang kết quả
    # Điều này sẽ phức tạp hơn và cần truy cập session hoặc query DB mới.

    # Xuất file PDF
    # File path can be tricky with tempfile on some systems, ensure permissions.
    try:
        # Create a temporary file that is automatically deleted after sending.
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            pdf_path = tmp.name
            pdf.output(pdf_path)
        
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'BaoCao_{round_info.round_name.replace(" ", "_")}.pdf'
            # Không dùng max_age=0 nếu muốn trình duyệt cache
        )
    except Exception as e:
        flash(f"Lỗi khi tạo file PDF: {e}", "danger")
        return redirect(url_for('round_detail', round_id=round_id)) # Hoặc một trang lỗi
    finally:
        if 'pdf_path' in locals() and os.path.exists(pdf_path): # Đảm bảo dọn dẹp file tạm
            try:
                os.unlink(pdf_path)
            except Exception as e_unlink:
                app.logger.error(f"Could not remove temp pdf file {pdf_path}: {e_unlink}")


@app.route('/history')
def history():
    try:
        rounds = RecruitmentRound.query.order_by(RecruitmentRound.created_at.desc()).all()
    except Exception as e:
        flash(f"Không thể tải lịch sử: {e}", "danger")
        rounds = []
    return render_template('history.html', rounds=rounds)

@app.route('/round/<int:round_id>')
def round_detail(round_id):
    round_info = RecruitmentRound.query.get_or_404(round_id)
    criteria_list_db = RecruitmentCriteria.query.filter_by(round_id=round_id).order_by(RecruitmentCriteria.criterion_id).all() # Sắp xếp để nhất quán
    criteria_matrix_db = CriteriaMatrix.query.filter_by(round_id=round_id).first()
    candidates_db = Candidate.query.filter_by(round_id=round_id).all()
    scores_db = CandidateScore.query.filter_by(round_id=round_id).order_by(CandidateScore.ranking).all()

    matrix_display_data = None
    weights_display = None
    criteria_names_display = [c.criterion_name for c in criteria_list_db] # Lấy tên từ DB làm chuẩn

    if criteria_matrix_db and criteria_matrix_db.matrix_data:
        try:
            # matrix_data từ DB là một dict
            matrix_display_data = criteria_matrix_db.matrix_data.get("matrix")
            weights_display = criteria_matrix_db.matrix_data.get("weights")
            # Có thể kiểm tra criteria_names_at_creation nếu muốn so sánh
        except Exception as e: # Xử lý nếu matrix_data không phải JSON hợp lệ (dù không nên)
            flash(f"Lỗi đọc dữ liệu ma trận: {e}", "warning")
            matrix_display_data = [] # Hoặc giá trị mặc định khác
            weights_display = []

    ranked_display = []
    for score_entry in scores_db:
        candidate_info = Candidate.query.get(score_entry.candidate_id)
        if candidate_info:
            ranked_display.append({
                'name': candidate_info.full_name, 
                'score': score_entry.total_score,
                'rank': score_entry.ranking
            })
    
    # Nếu muốn hiển thị cả các ma trận so sánh ứng viên, bạn cần query từ DB (nếu đã lưu)
    # hoặc thiết kế lại cách lưu/truy xuất. Hiện tại, chúng không được lưu vào DB trong logic đã cung cấp.

    return render_template('round_detail.html', 
                           round=round_info, 
                           criteria_names=criteria_names_display, 
                           matrix_data=matrix_display_data, # Chỉ truyền phần ma trận
                           weights_criteria=weights_display, # Truyền trọng số
                           consistency_ratio_criteria=criteria_matrix_db.consistency_ratio if criteria_matrix_db else None,
                           candidates=candidates_db, 
                           ranked_scores=ranked_display)


@app.route('/delete_round/<int:round_id>')
def delete_round(round_id):
    try:
        # Đảm bảo xóa theo đúng thứ tự khóa ngoại
        CandidateScore.query.filter_by(round_id=round_id).delete()
        # Nếu có bảng CandidateCriterionMatrices, xóa ở đây
        CriteriaMatrix.query.filter_by(round_id=round_id).delete()
        Candidate.query.filter_by(round_id=round_id).delete()
        RecruitmentCriteria.query.filter_by(round_id=round_id).delete()
        RecruitmentRound.query.filter_by(round_id=round_id).delete()
        db.session.commit()
        flash("Đợt tuyển dụng và tất cả dữ liệu liên quan đã được xóa.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Lỗi khi xóa đợt tuyển dụng: {str(e)}", "danger")
    return redirect(url_for('history'))

if __name__ == '__main__':
    # Để tạo bảng trong DB lần đầu, bỏ comment dòng dưới và chạy app một lần, sau đó comment lại.
    # with app.app_context():
    # db.create_all()
    app.run(debug=True)
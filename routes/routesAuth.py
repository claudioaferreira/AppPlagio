# routes/routesAuth.py
from flask import Blueprint, render_template, request, redirect, url_for, session, flash

# Usamos 'auth' como nombre del blueprint y le decimos dónde están sus plantillas
auth_bp = Blueprint('auth_bp', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    # Si el usuario ya está logueado, lo mandamos al inicio
    if 'user_id' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # --- LÓGICA DE VALIDACIÓN (Ejemplo) ---
        # Aquí deberías comprobar contra tu base de datos
        # Este es solo un ejemplo simple:
        if username == 'admin' and password == '1234':
            # Guardamos el ID del usuario en la sesión
            session['user_id'] = 1 # O el ID real del usuario
            session['username'] = username
            flash('¡Bienvenido de nuevo!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Usuario o contraseña incorrectos.', 'danger')
            return redirect(url_for('auth_bp.login'))
            
    # Si es GET, solo mostramos la página de login
    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    # Limpiamos la sesión
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Has cerrado sesión exitosamente.', 'info')
    return redirect(url_for('home'))
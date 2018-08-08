from flask_wtf import FlaskForm
from wtforms import FloatField, StringField
from wtforms.validators import DataRequired, optional, NumberRange

class SignUp(FlaskForm):
    name = StringField('Full Name', validators=[DataRequired()])
    phone = StringField('Phone', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    occupation = StringField('Occupation', validators=[DataRequired()])
    age = FloatField('Age', validators=[DataRequired()])
    description = StringField('You in 3 words', validators=[DataRequired()])
    unique_id = StringField('Unique ID', validators=[DataRequired()])
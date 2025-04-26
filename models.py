from datetime import datetime
from app import db
from flask_login import UserMixin
import json

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    role = db.Column(db.String(20), default='analyst')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

class DataSource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    source_type = db.Column(db.String(50), nullable=False)  # twitter, telegram, etc.
    config = db.Column(db.Text)  # JSON configuration for the source
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_ingestion = db.Column(db.DateTime)

class DetectedNarrative(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text, nullable=False)  # Changed from String(200) to Text to handle longer titles
    description = db.Column(db.Text)
    confidence_score = db.Column(db.Float)
    first_detected = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='active')  # active, archived, debunked
    language = db.Column(db.String(10), default='en')
    vector_id = db.Column(db.String(100))  # Reference to vector in FAISS

class NarrativeInstance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    narrative_id = db.Column(db.Integer, db.ForeignKey('detected_narrative.id'), nullable=False)
    source_id = db.Column(db.Integer, db.ForeignKey('data_source.id'))
    content = db.Column(db.Text, nullable=False)
    meta_data = db.Column(db.Text)  # JSON with source-specific metadata
    url = db.Column(db.String(1024))
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)
    evidence_hash = db.Column(db.String(256))  # Hash for immutable evidence storage
    
    narrative = db.relationship('DetectedNarrative', backref='instances')
    source = db.relationship('DataSource')

class BeliefNode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    node_type = db.Column(db.String(50))  # claim, entity, source, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    meta_data = db.Column(db.Text)  # JSON with additional metadata

class BeliefEdge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('belief_node.id'), nullable=False)
    target_id = db.Column(db.Integer, db.ForeignKey('belief_node.id'), nullable=False)
    relation_type = db.Column(db.String(50))  # supports, contradicts, mentions, etc.
    weight = db.Column(db.Float, default=1.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    source_node = db.relationship('BeliefNode', foreign_keys=[source_id])
    target_node = db.relationship('BeliefNode', foreign_keys=[target_id])

class CounterMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    narrative_id = db.Column(db.Integer, db.ForeignKey('detected_narrative.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    strategy = db.Column(db.String(100))  # fact-checking, prebunking, etc.
    status = db.Column(db.String(50), default='draft')  # draft, approved, deployed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    approved_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    narrative = db.relationship('DetectedNarrative')
    creator = db.relationship('User', foreign_keys=[created_by])
    approver = db.relationship('User', foreign_keys=[approved_by])

class SystemLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    log_type = db.Column(db.String(50))  # info, warning, error, security
    component = db.Column(db.String(100))  # detector, analyzer, ingestion, etc.
    message = db.Column(db.Text)
    meta_data = db.Column(db.Text)  # JSON with additional context
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}

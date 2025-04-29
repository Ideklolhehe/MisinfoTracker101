from datetime import datetime
from app import db
from flask_login import UserMixin
from flask_dance.consumer.storage.sqla import OAuthConsumerMixin
from sqlalchemy import UniqueConstraint, Text, JSON, Float
import json
import enum
import uuid

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(256), nullable=True)
    role = db.Column(db.String(20), default='analyst')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    first_name = db.Column(db.String(64), nullable=True)
    last_name = db.Column(db.String(64), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    profile_image_url = db.Column(db.String(512), nullable=True)
    meta_data = db.Column(db.Text)  # JSON with user preferences, settings, expertise areas, etc.
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}
        
class OAuth(OAuthConsumerMixin, db.Model):
    user_id = db.Column(db.String, db.ForeignKey(User.id))
    browser_session_key = db.Column(db.String, nullable=False)
    user = db.relationship(User)

    __table_args__ = (UniqueConstraint(
        'user_id',
        'browser_session_key',
        'provider',
        name='uq_user_browser_session_key_provider',
    ),)

class DataSource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    source_type = db.Column(db.String(50), nullable=False)  # twitter, telegram, etc.
    config = db.Column(db.Text)  # JSON configuration for the source
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_ingestion = db.Column(db.DateTime)
    meta_data = db.Column(db.Text)  # JSON with additional source metadata (credibility, topics, etc.)
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}

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
    meta_data = db.Column(db.Text)  # JSON with narrative-specific metadata
    
    # Relationships
    counter_messages = db.relationship('CounterMessage', backref='narrative', lazy='dynamic')
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}
    
    def to_dict(self):
        """Return a dictionary representation of the narrative."""
        meta_data = self.get_meta_data()
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'confidence_score': self.confidence_score,
            'first_detected': self.first_detected.isoformat() if self.first_detected else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'status': self.status,
            'language': self.language,
            'complexity_score': meta_data.get('complexity_score'),
            'propagation_score': meta_data.get('propagation_score'),
            'threat_score': meta_data.get('threat_score'),
            'source_count': meta_data.get('source_count'),
            'stream_cluster': meta_data.get('stream_cluster'),
            'temporal_cluster': meta_data.get('temporal_cluster')
        }

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
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}

class BeliefNode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    node_type = db.Column(db.String(50))  # claim, entity, source, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    meta_data = db.Column(db.Text)  # JSON with additional metadata
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}

class BeliefEdge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('belief_node.id'), nullable=False)
    target_id = db.Column(db.Integer, db.ForeignKey('belief_node.id'), nullable=False)
    relation_type = db.Column(db.String(50))  # supports, contradicts, mentions, etc.
    weight = db.Column(db.Float, default=1.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    meta_data = db.Column(db.Text)  # JSON with additional relationship metadata
    
    source_node = db.relationship('BeliefNode', foreign_keys=[source_id])
    target_node = db.relationship('BeliefNode', foreign_keys=[target_id])
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}

class CounterMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    narrative_id = db.Column(db.Integer, db.ForeignKey('detected_narrative.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    dimension = db.Column(db.String(50))  # emotional, cognitive, inoculation
    strategy = db.Column(db.String(50))  # Specific strategy used for counter-narrative
    status = db.Column(db.String(50), default='draft')  # draft, approved, deployed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.String, db.ForeignKey('users.id'))
    approved_by = db.Column(db.String, db.ForeignKey('users.id'))
    meta_data = db.Column(db.Text)  # JSON with effectiveness metrics, audience data, etc.
    
    # Relationships defined without circular references
    creator = db.relationship('User', foreign_keys=[created_by])
    approver = db.relationship('User', foreign_keys=[approved_by])
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}
    
    def to_dict(self):
        """Return a dictionary representation of the counter-message."""
        meta_data = self.get_meta_data()
        effectiveness = meta_data.get('effectiveness', {})
        
        return {
            'id': self.id,
            'narrative_id': self.narrative_id,
            'content': self.content,
            'dimension': self.dimension,
            'strategy': self.strategy,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'created_by': self.created_by,
            'approved_by': self.approved_by,
            'exposure': effectiveness.get('exposure', 0),
            'engagement': effectiveness.get('engagement', 0),
            'sentiment_shift': effectiveness.get('sentiment_shift', 0),
            'reach': effectiveness.get('reach', 0)
        }

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


class AdversarialContent(db.Model):
    """Model for storing adversarial content generated for system training."""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text, nullable=False)
    content = db.Column(db.Text, nullable=False)
    topic = db.Column(db.String(50), nullable=False)  # health, politics, etc.
    misinfo_type = db.Column(db.String(50), nullable=False)  # conspiracy_theory, misleading_statistics, etc.
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    generation_method = db.Column(db.String(20), default='ai')  # ai, template, human
    meta_data = db.Column(db.Text)  # JSON with additional information
    variant_of_id = db.Column(db.Integer, db.ForeignKey('adversarial_content.id'), nullable=True)
    is_active = db.Column(db.Boolean, default=True)  # Whether to use in training
    
    # Relationships
    variants = db.relationship('AdversarialContent', 
                              backref=db.backref('parent', remote_side=[id]),
                              lazy='dynamic')
    evaluations = db.relationship('AdversarialEvaluation', backref='content', lazy='dynamic')
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}


class AdversarialEvaluation(db.Model):
    """Model for storing evaluation results from adversarial training content."""
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('adversarial_content.id'), nullable=False)
    detector_version = db.Column(db.String(50))  # Version of detector used


class MisinformationEvent(db.Model):
    """Model for tracking misinformation events for source reliability analysis."""
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('data_source.id'), nullable=False)
    narrative_id = db.Column(db.Integer, db.ForeignKey('detected_narrative.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    reporter_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=True)
    evaluator_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=True)
    confidence = db.Column(db.Float, default=1.0)  # Confidence in the misinformation assessment
    correct_detection = db.Column(db.Boolean, nullable=True)  # Whether system correctly identified it as misinfo
    evaluation_date = db.Column(db.DateTime, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    meta_data = db.Column(db.Text)  # JSON with additional context about the event
    
    # Relationships
    source = db.relationship('DataSource', backref='misinfo_events')
    narrative = db.relationship('DetectedNarrative', backref='misinfo_events')
    reporter = db.relationship('User', foreign_keys=[reporter_id], backref='reported_misinfo_events')
    evaluator = db.relationship('User', foreign_keys=[evaluator_id], backref='evaluated_misinfo_events')
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}


class EvidenceRecord(db.Model):
    """Model for storing immutable evidence records."""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    content_hash = db.Column(db.String(256), nullable=False)  # Hash of the content
    source_url = db.Column(db.String(1024))  # Original source URL
    capture_date = db.Column(db.DateTime, default=datetime.utcnow)
    content_type = db.Column(db.String(100))  # MIME type
    content_data = db.Column(db.Text, nullable=True)  # Actual content data (if text)
    verified = db.Column(db.Boolean, default=False)
    verification_method = db.Column(db.String(50), nullable=True)  # Method used for verification
    meta_data = db.Column(db.Text)  # JSON with additional metadata
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}


class PublishedContent(db.Model):
    """Model for tracking content published to decentralized networks."""
    id = db.Column(db.Integer, primary_key=True)
    content_type = db.Column(db.String(50), nullable=False)  # narrative_analysis, counter_narrative, evidence_record, etc.
    reference_id = db.Column(db.String(36), nullable=False)  # ID of the referenced content in the respective table
    ipfs_hash = db.Column(db.String(256), nullable=False)  # IPFS content identifier
    ipns_name = db.Column(db.String(256), nullable=True)  # IPNS name (if published to IPNS)
    publisher_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=True)
    status = db.Column(db.String(50), default='published')  # published, revoked, updated
    title = db.Column(db.String(256))
    description = db.Column(db.Text)
    publication_date = db.Column(db.DateTime, default=datetime.utcnow)
    meta_data = db.Column(db.Text)  # JSON with additional metadata
    
    # Relationships
    publisher = db.relationship('User', backref='published_content')
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}


class ContentType(enum.Enum):
    """Enum for types of user-submitted content."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    TEXT_IMAGE = "text_image"
    TEXT_VIDEO = "text_video"


class VerificationType(enum.Enum):
    """Enum for types of verification performed."""
    MISINFORMATION = "misinformation"
    AI_GENERATED = "ai_generated"
    AUTHENTICITY = "authenticity"
    FACTUAL_ACCURACY = "factual_accuracy"


class VerificationStatus(enum.Enum):
    """Enum for status of verification process."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UserSubmission(db.Model):
    """Model for storing user-submitted content for verification."""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    description = db.Column(db.Text)
    content_type = db.Column(db.String(20), nullable=False)  # text, image, video, text_image, text_video
    text_content = db.Column(db.Text)
    media_path = db.Column(db.String(500))  # Path to stored image/video
    source_url = db.Column(db.String(1024))  # Original source if provided
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))  # To track submission source
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=True)  # Optional, for logged-in users
    meta_data = db.Column(db.Text)  # JSON with additional submission metadata
    
    # Relationships
    user = db.relationship('User', backref='submissions')
    verification_results = db.relationship('VerificationResult', backref='submission', lazy='dynamic')
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}


class VerificationResult(db.Model):
    """Model for storing verification results for user-submitted content."""
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('user_submission.id'), nullable=False)
    verification_type = db.Column(db.String(50), nullable=False)  # misinformation, ai_generated, authenticity, factual_accuracy
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    confidence_score = db.Column(db.Float)  # 0-1 confidence in the verification result
    result_summary = db.Column(db.Text)  # Human-readable summary of the result
    evidence = db.Column(db.Text)  # Supporting evidence for the verification
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    meta_data = db.Column(db.Text)  # JSON with detailed verification results
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}


class SystemCredential(db.Model):
    """Model for storing system API credentials securely."""
    id = db.Column(db.Integer, primary_key=True)
    credential_type = db.Column(db.String(50), unique=True, nullable=False)  # openai, youtube, twitter, telegram, dark_web
    credential_data = db.Column(db.Text, nullable=False)  # JSON with encrypted credential data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_data = db.Column(db.Text)  # JSON with additional metadata
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}


class ContentItem(db.Model):
    """Model for storing content items collected from various sources."""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text)
    content = db.Column(db.Text, nullable=False)
    source = db.Column(db.String(100))  # Source name or type
    url = db.Column(db.String(1024))
    published_date = db.Column(db.DateTime)
    content_type = db.Column(db.String(50))  # web, social, news, etc.
    is_processed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_data = db.Column(db.Text)  # JSON with additional metadata
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}


class WebSource(db.Model):
    """Model for web scraping data sources."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    url = db.Column(db.String(1024), nullable=False)
    source_type = db.Column(db.String(50), nullable=False)  # web_page, web_crawl, web_search
    is_active = db.Column(db.Boolean, default=True)
    config = db.Column(db.Text)  # JSON configuration for the source
    meta_data = db.Column(db.Text)  # JSON with additional source metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_ingestion = db.Column(db.DateTime)
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}
    
    def set_config(self, data):
        self.config = json.dumps(data)
    
    def get_config(self):
        return json.loads(self.config) if self.config else {}


class FocusedDomain(db.Model):
    """Model for domains to focus on for monitoring."""
    id = db.Column(db.Integer, primary_key=True)
    domain = db.Column(db.String(255), unique=True, nullable=False)
    category = db.Column(db.String(50))  # news, social, government, etc.
    priority = db.Column(db.Integer, default=2)  # 1=high, 2=medium, 3=low
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_check = db.Column(db.DateTime)
    meta_data = db.Column(db.Text)  # JSON with additional metadata
    created_by = db.Column(db.String, db.ForeignKey('users.id'), nullable=True)
    
    # Relationships
    creator = db.relationship('User', foreign_keys=[created_by])
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}


class SearchTerm(db.Model):
    """Model for search terms to monitor for content."""
    id = db.Column(db.Integer, primary_key=True)
    term = db.Column(db.String(255), unique=True, nullable=False)
    category = db.Column(db.String(50))  # misinformation, health, politics, etc.
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_check = db.Column(db.DateTime)
    meta_data = db.Column(db.Text)  # JSON with additional metadata
    created_by = db.Column(db.String, db.ForeignKey('users.id'), nullable=True)
    
    # Relationships
    creator = db.relationship('User', foreign_keys=[created_by])
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}


# Note: WebSource is intentionally removed from here, as it's defined earlier in the file at line ~386


class WebSourceJobStatus(enum.Enum):
    """Enum for web source job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WebSourceJob(db.Model):
    """Model for tracking web source ingestion jobs."""
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('web_source.id'), nullable=False)
    job_type = db.Column(db.String(50))  # scan, crawl, search, monitor
    status = db.Column(db.String(20), default=WebSourceJobStatus.PENDING.value)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    job_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True)
    results = db.Column(db.Text)  # JSON with job results
    error_message = db.Column(db.Text)
    meta_data = db.Column(db.Text)  # JSON with additional job metadata
    created_by = db.Column(db.String, db.ForeignKey('users.id'), nullable=True)
    
    # Relationships
    creator = db.relationship('User', foreign_keys=[created_by])
    
    def set_meta_data(self, data):
        self.meta_data = json.dumps(data)
    
    def get_meta_data(self):
        return json.loads(self.meta_data) if self.meta_data else {}
    
    def set_results(self, data):
        self.results = json.dumps(data)
    
    def get_results(self):
        return json.loads(self.results) if self.results else {}

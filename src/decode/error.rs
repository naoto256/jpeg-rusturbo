//! Decode-side error type. Modeled after the encoder's `io::Result`
//! shape but with structured variants for diagnostic-friendly
//! failures.

use std::fmt;
use std::io;

#[derive(Debug)]
pub enum DecodeError {
    /// Hit end of buffer before completing a structural read.
    UnexpectedEof,
    /// Bytes don't conform to the JPEG syntax (bad marker, length
    /// mismatch, etc.). The `&'static str` is a short reason tag.
    Malformed(&'static str),
    /// The stream is valid JPEG but uses a feature we don't decode
    /// (arithmetic coding, hierarchical mode, 12-bit precision …).
    Unsupported(&'static str),
    /// Image dimensions or component layout we don't accept (overflow,
    /// zero dimensions, unsupported component count).
    InvalidDimensions(&'static str),
    /// I/O failure from the caller-supplied sink (used by the trait
    /// adapter into `image`).
    Io(io::Error),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnexpectedEof => f.write_str("unexpected end of JPEG data"),
            DecodeError::Malformed(why) => write!(f, "malformed JPEG: {why}"),
            DecodeError::Unsupported(what) => write!(f, "unsupported JPEG feature: {what}"),
            DecodeError::InvalidDimensions(why) => write!(f, "invalid dimensions: {why}"),
            DecodeError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DecodeError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for DecodeError {
    fn from(e: io::Error) -> Self {
        DecodeError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, DecodeError>;

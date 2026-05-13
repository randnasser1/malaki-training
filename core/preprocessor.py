"""
Polars-based text preprocessing module with checkpointing.
Provides efficient text cleaning for large datasets.
"""

import os
import pandas as pd
import polars as pl
from typing import Union, Optional, List, Tuple

# Import from your existing modules
from core.cleaning import clean_text


class PolarsPreprocessor:
    """
    Polars-based text preprocessing with checkpointing.
    
    This class provides efficient text cleaning for large datasets using
    Polars for high performance. It includes checkpointing to save
    intermediate results and avoid reprocessing.
    
    Attributes:
        slang_dict (dict, optional): Dictionary for slang expansion
    """
    
    def __init__(self, slang_dict: dict = None):
        """
        Initialize the preprocessor.
        
        Args:
            slang_dict: Dictionary mapping slang terms to their expansions
        """
        self.slang_dict = slang_dict
    
    def preprocess(self, 
                   df: Union[pd.DataFrame, pl.DataFrame],
                   text_column: str = 'text',
                   metadata_columns: Optional[List[str]] = None,
                   aggressive: bool = False,
                   checkpoint_dir: Optional[str] = None,
                   stage_name: Optional[str] = None,
                   min_text_length: int = 10,
                   return_indices: bool = False) -> Union[pd.DataFrame, pl.DataFrame, Tuple]:
        """
        Preprocess DataFrame with optional checkpointing.
        
        This method cleans text data by applying the clean_text function,
        filters out empty/short texts, and removes texts that are mostly
        non-alphanumeric characters.
        
        Args:
            df: Input DataFrame (pandas or polars)
            text_column: Name of the column containing text to clean
            metadata_columns: List of column names to preserve alongside text
            aggressive: Whether to apply aggressive cleaning
            checkpoint_dir: Directory to save/load checkpoints
            stage_name: Name for checkpoint file
            min_text_length: Minimum character length for texts to keep
            return_indices: If True, returns (processed_df, kept_indices)
            
        Returns:
            Processed DataFrame (same type as input) or tuple with indices
        """
        # Check for checkpoint
        if checkpoint_dir and stage_name:
            checkpoint_path = f"{checkpoint_dir}/{stage_name}.parquet"
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint: {stage_name}")
                df_result = pl.read_parquet(checkpoint_path)
                if return_indices:
                    # Can't recover indices from checkpoint
                    return df_result, None
                return df_result
        
        # Store original indices
        if isinstance(df, pd.DataFrame):
            original_indices = list(range(len(df)))
        else:
            original_indices = list(range(df.height))
        
        # Convert pandas to polars if needed, preserving index
        return_pandas = False
        if isinstance(df, pd.DataFrame):
            # Add index as a column
            df_with_index = df.copy()
            df_with_index['_original_index'] = original_indices
            polars_df = pl.from_pandas(df_with_index)
            return_pandas = True
        else:
            # For polars, add index column
            polars_df = df.with_columns(
                pl.Series('_original_index', original_indices)
            )
        
        # Store metadata separately if specified
        if metadata_columns:
            metadata_df = polars_df.select(metadata_columns)
        
        total = polars_df.height
        print(f"Cleaning {total} texts...")
        
        # Apply cleaning using map_elements
        polars_df = polars_df.with_columns(
            pl.col(text_column).map_elements(
                lambda x: clean_text(
                    x, 
                    aggressive=aggressive, 
                    slang_dict=self.slang_dict
                ),
                return_dtype=pl.String
            ).alias(text_column)
        )
        
        # Filter empty and short texts
        polars_df = polars_df.filter(
            pl.col(text_column).str.strip_chars() != ""
        )
        polars_df = polars_df.filter(
            pl.col(text_column).str.len_chars() >= min_text_length
        )
        
        # Filter mostly non-alphanumeric
        def is_mostly_text(text: str) -> bool:
            """Check if text is mostly alphanumeric characters."""
            if not text or not isinstance(text, str) or len(text) == 0:
                return False
            alnum_spaces = sum(c.isalnum() or c.isspace() for c in text)
            return alnum_spaces / len(text) > 0.5
        
        # Apply the filter
        mask = polars_df.select(
            pl.col(text_column).map_elements(
                is_mostly_text, 
                return_dtype=pl.Boolean
            ).alias("mask")
        ).to_series()
        
        polars_df = polars_df.filter(mask)
        
        # If we had metadata, apply the same filtering and add back
        if metadata_columns:
            # Filter metadata using the same mask
            metadata_df = metadata_df.filter(mask)
            # Add metadata back
            for col in metadata_columns:
                if col != text_column:  # Avoid duplicating text column
                    polars_df = polars_df.with_columns(
                        metadata_df[col].alias(col)
                    )
        
        # Get kept indices
        kept_indices = polars_df['_original_index'].to_list()
        
        # Remove the index column before returning
        polars_df = polars_df.drop('_original_index')
        
        final_count = polars_df.height
        
        # Print report
        print(f"\nCleaning Report:")
        print(f"- Original: {total} texts")
        print(f"- Removed: {total - final_count} texts")
        print(f"- Final dataset: {final_count} samples")
        print(f"- Retention rate: {final_count/total*100:.1f}%")
        
        # Save checkpoint if requested
        if checkpoint_dir and stage_name:
            os.makedirs(checkpoint_dir, exist_ok=True)
            polars_df.write_parquet(checkpoint_path)
            print(f"Saved checkpoint: {stage_name}")
        
        # Convert back to pandas if that's what came in
        result = polars_df.to_pandas() if return_pandas else polars_df
        
        if return_indices:
            return result, kept_indices
        return result
    
   
    
    def clean_text_only(self,
                       texts: List[str],
                       aggressive: bool = False) -> List[str]:
        """
        Clean a list of texts without DataFrame operations.
        
        Useful for quick cleaning of small text lists.
        
        Args:
            texts: List of text strings to clean
            aggressive: Whether to apply aggressive cleaning
            
        Returns:
            List of cleaned texts
        """
        return [
            clean_text(text, aggressive=aggressive, slang_dict=self.slang_dict)
            for text in texts
        ]


# Optional: Add a function to get a preconfigured preprocessor
def get_preprocessor(use_slang: bool = True) -> PolarsPreprocessor:
    """
    Get a configured PolarsPreprocessor instance.
    
    Args:
        use_slang: Whether to include slang dictionary
        
    Returns:
        Configured preprocessor
    """
    from core.cleaning import SLANG_DICT
    slang_dict = SLANG_DICT if use_slang else None
    return PolarsPreprocessor(slang_dict=slang_dict)
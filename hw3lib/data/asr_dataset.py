from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H3Tokenizer

'''
TODO: Implement this class.

Specification:
The ASRDataset class provides data loading and processing for ASR (Automatic Speech Recognition):

1. Data Organization:
   - Handles dataset partitions (train-clean-100, dev-clean, test-clean)
   - Features stored as .npy files in fbank directory
   - Transcripts stored as .npy files in text directory
   - Maintains alignment between features and transcripts

2. Feature Processing:
   - Loads log mel filterbank features from .npy files
   - Supports multiple normalization strategies:
     * global_mvn: Global mean and variance normalization
     * cepstral: Per-utterance mean and variance normalization
     * none: No normalization
   - Applies SpecAugment data augmentation during training:
     * Time masking: Masks random time steps
     * Frequency masking: Masks random frequency bands

3. Transcript Processing:
   - Creates shifted (SOS-prefixed) and golden (EOS-suffixed) versions
   - Tracks statistics for perplexity calculation
   - Handles tokenization using H3Tokenizer

4. Batch Preparation:
   - Pads features and transcripts to batch-uniform lengths
   - Provides lengths for packed sequence processing
   - Ensures proper device placement and tensor types

Key Requirements:
- Must maintain feature-transcript alignment
- Must handle variable-length sequences
- Must track maximum lengths for both features and text
- Must implement proper padding for batching
- Must apply SpecAugment only during training
- Must support different normalization strategies
'''

class ASRDataset(Dataset):
    def __init__(
            self,
            partition:Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config:dict,
            tokenizer:H3Tokenizer,
            isTrainPartition:bool,
            global_stats:Optional[Tuple[torch.Tensor, torch.Tensor]]=None
    ):
        """
        Initialize the ASRDataset for ASR training/validation/testing.
        Args:
            partition (str): Dataset partition ('train-clean-100', 'dev-clean', or 'test-clean')
            config (dict): Configuration dictionary containing dataset settings
            tokenizer (H3Tokenizer): Tokenizer for encoding/decoding text
            isTrainPartition (bool): Whether this is the training partition
                                     Used to determine if SpecAugment should be applied.
            global_stats (tuple, optional): (mean, std) computed from training set.
                                          If None and using global_mvn, will compute during loading.
                                          Should only be None for training set.
                                          Should be provided for dev and test sets.
        """
        # Store basic configuration
        self.config    = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # Get tokenizer ids for special tokens (eos, sos, pad)
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # Set up data paths
        # Use root and partition to get the feature directory
        self.fbank_dir   = os.path.join(config['root'], partition, 'fbank')

        # Get all feature files in the feature directory in sorted order
        # Store bare file names (not full paths) so fbank and text names can be compared directly
        self.fbank_files = sorted(f for f in os.listdir(self.fbank_dir) if f.endswith('.npy'))

        # Take subset
        subset_size      = int(config['subset'] * len(self.fbank_files))
        self.fbank_files = self.fbank_files[:subset_size]

        # Get the number of samples in the dataset
        self.length      = len(self.fbank_files)

        # Case on partition.
        # Why will test-clean need to be handled differently?
        if self.partition != "test-clean":
            # Use root and partition to get the text directory
            self.text_dir   = os.path.join(config['root'], partition, 'text')

            # Get all text files in the text directory in sorted order
            self.text_files = sorted(f for f in os.listdir(self.text_dir) if f.endswith('.npy'))

            # Take subset
            self.text_files = self.text_files[:subset_size]
            
            # Verify data alignment
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        # Initialize lists to store features and transcripts
        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []
        
        # Initialize counters for character and token counts
        # DO NOT MODIFY
        self.total_chars  = 0
        self.total_tokens = 0
        
        # Initialize max length variables
        # DO NOT MODIFY
        self.feat_max_len = 0
        self.text_max_len = 0
        
        # Initialize Welford's algorithm accumulators if needed for global_mvn
        # DO NOT MODIFY
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions when using global_mvn")
            count = 0
            mean = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2 = torch.zeros(self.config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # Load features
            # Features are of shape (num_feats, time)
            feat = np.load(os.path.join(self.fbank_dir, self.fbank_files[i]))

            # Truncate features to num_feats set by you in the config
            feat = feat[:self.config['num_feats'], :]

            # Append to self.feats (num_feats is set by you in the config)
            # Optionally keep features in RAM as float16 (config 'feat_dtype') to halve memory:
            # train-clean-100 is ~11.6 GB as float32. __getitem__ casts back to float32.
            self.feats.append(feat.astype(np.float16) if self.config.get('feat_dtype', 'float32') == 'float16' else feat)

            # Track max length (time dimension)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            # Update global statistics if needed (DO NOT MODIFY)
            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.FloatTensor(feat)  # (num_feats, time)
                batch_count = feat_tensor.shape[1]     # number of time steps
                count += batch_count
                
                # Update mean and M2 for all time steps at once
                delta = feat_tensor - mean.unsqueeze(1)  # (num_feats, time)
                mean += delta.sum(dim=1) / count         # (num_feats,)
                delta2 = feat_tensor - mean.unsqueeze(1) # (num_feats, time)
                M2 += (delta * delta2).sum(dim=1)        # (num_feats,)

            if self.partition != "test-clean":
                # Load the transcript
                # Important Note: This is a very important line of code and you should check whether your transcript is correct after loading (and very dependent in evaluation)
                # Transcripts are stored as an array of single characters, so join them back into a string
                transcript = ''.join(np.load(os.path.join(self.text_dir, self.text_files[i])))

                # Track character count (before tokenization)
                self.total_chars += len(transcript)

                # Use tokenizer to encode the transcript (see tokenizer.encode for details)
                tokenized = self.tokenizer.encode(transcript)

                # Track token count (excluding special tokens)
                # DO NOT MODIFY
                self.total_tokens += len(tokenized)

                # Track max length (add 1 for the sos/eos tokens)
                # DO NOT MODIFY
                self.text_max_len = max(self.text_max_len, len(tokenized)+1)
                
                # Create shifted and golden versions by adding sos and eos tokens
                self.transcripts_shifted.append([self.sos_token] + tokenized)
                self.transcripts_golden.append(tokenized + [self.eos_token])


        # Calculate average characters per token
        # DO NOT MODIFY 
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        
        if self.partition != "test-clean":
            # Verify data alignment
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # Compute final global statistics if needed
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                # Compute variance and standard deviation
                variance = M2/(count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # Initialize SpecAugment transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        '''
        Get the average number of characters per token. Used to calculate character-level perplexity.
        DO NOT MODIFY
        '''
        return self.avg_chars_per_token

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        DO NOT MODIFY
        """
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (features, shifted_transcript, golden_transcript) where:
                - features: FloatTensor of shape (num_feats, time)
                - shifted_transcript: LongTensor (time) or None
                - golden_transcript: LongTensor  (time) or None
        """
        # Load features
        feat = torch.from_numpy(self.feats[idx]).float() # (num_feats, time), float32 even if stored as float16

        # Apply normalization
        if self.config['norm'] == 'global_mvn':
            assert self.global_mean is not None and self.global_std is not None, "Global mean and std must be computed before normalization"
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)
        elif self.config['norm'] == 'none':
            pass
        
        # Get transcripts for non-test partitions
        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            # Get transcripts for non-test partitions
            shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
            golden_transcript  = torch.LongTensor(self.transcripts_golden[idx])

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate and pad a batch of samples to create a batch of fixed-length padded features and transcripts.

        Args:
            batch (list): List of samples from __getitem__

        Returns:
            tuple: (padded_features, padded_shifted, padded_golden, feat_lengths, transcript_lengths) where:
                - padded_features: Tensor of shape (batch, max_time, num_feats)
                - padded_shifted: Tensor of shape (batch, max_len) or None
                - padded_golden: Tensor of shape (batch, max_len) or None  
                - feat_lengths: Tensor of original feature lengths of shape (batch)
                - transcript_lengths: Tensor of transcript lengths of shape (batch) or None
        """
        # Collect transposed features from the batch into a list of tensors (B x T x F)
        batch_feats  = [feat.transpose(0, 1) for feat, _, _ in batch]

        # Collect feature lengths from the batch into a tensor
        feat_lengths = torch.LongTensor([feat.shape[0] for feat in batch_feats]) # B

        # Pad features to create a batch of fixed-length padded features
        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=0.0) # B x T x F

        # Handle transcripts for non-test partitions
        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            # Collect shifted and golden transcripts from the batch into a list of tensors (B x T)
            batch_shifted      = [shifted for _, shifted, _ in batch] # B x T
            batch_golden       = [golden for _, _, golden in batch]   # B x T

            # Collect transcript lengths from the batch into a tensor
            transcript_lengths = torch.LongTensor([len(shifted) for shifted in batch_shifted]) # B

            # Pad transcripts to create a batch of fixed-length padded transcripts
            padded_shifted     = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token) # B x T
            padded_golden      = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)  # B x T

        # Apply SpecAugment for training
        if self.config["specaug"] and self.isTrainPartition:
            # Permute the features to (B x F x T)
            padded_feats = padded_feats.permute(0, 2, 1) # B x F x T

            # Apply frequency masking
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)

            # Apply time masking
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)

            # Permute the features back to (B x T x F)
            padded_feats = padded_feats.permute(0, 2, 1) # B x T x F

        # Return the padded features, padded shifted, padded golden, feature lengths, and transcript lengths
        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths


import type { ChangeEvent, ReactElement } from 'react';

export type UploadProps = {
  onFileSelect?: (file: File) => void;
};

function Upload({ onFileSelect }: UploadProps): ReactElement {
  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && onFileSelect) {
      onFileSelect(file);
    }
  };

  return <input type="file" accept=".dcm,.nii,.nii.gz" onChange={handleChange} />;
}

export default Upload;

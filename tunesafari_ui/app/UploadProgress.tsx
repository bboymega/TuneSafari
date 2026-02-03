export default function UploadProgress ({ state }: { state: string }) {
    return (
    <div
        id="uploadProgress"
        style={{
        position: 'fixed',
        top: '50%',
        transform: 'translateY(-50%)',
        fontFamily: '"OPTICopperplate-Light", sans-serif',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        color: '#fff',
        padding: '10px 20px',
        borderRadius: '5px',
        fontSize: '2rem',
        zIndex: 1000,
        textAlign: 'center',
        margin: '0 auto',
        maxWidth: "clamp(0px, 90vw, 360px)",
        }}
    >
        <div
        id="spinner"
        className="mx-auto"
        style={{
            width: '180px',
            height: '150px',
            backgroundImage: "url('assets/img/loading.gif')",
            backgroundSize: 'cover',
            margin: '0 auto',
        }}
        />
        <div id="uploadProgressText" style={{ marginTop: '1rem', fontSize: '1.8rem' }}>
        {state}
        </div>
    </div>
    );
}
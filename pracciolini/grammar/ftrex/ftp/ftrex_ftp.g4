/**
 * @file ftrex_ftp.g4
 * @brief This grammar defines the structure for parsing ftrex_ftp files.
 *
 * The ftrex_ftp grammar is designed to parse files that consist of multiple sections,
 * including tree definitions, process commands, import commands, and metadata.
 * Each section is defined with specific rules that dictate the expected format and content.
 */

grammar ftrex_ftp;

/**
 * @brief Represents the entire file content.
 *
 * The file consists of zero or more sections followed by an EOF (End of File) symbol.
 */
file_
    : section* EOF
    ;

/**
 * @brief Defines a section in the file.
 *
 * A section can be one of the following types: treeSection, processSection,
 * importSection, limitSection, or metaArgs.
 */
section
    : treeSection
    | processSection
    | importSection
    | limitSection
    | metaArgs
    ;

/**
 * @brief Describes a tree section containing gates and ends with 'ENDTREE'.
 *
 * Each treeSection consists of zero or more gates followed by the keyword 'ENDTREE' and an end of line.
 */
treeSection
    : gate* 'ENDTREE' EOL
    ;

/**
 * @brief Represents a gate within a tree section.
 *
 * A gate is defined by a gate identifier, gate type, one or more child references, and an end of line.
 */
gate
    : gateId gateType childRef+ EOL
    ;

/**
 * @brief Defines a process section starting with 'PROCESS'.
 *
 * The process section includes a list of process commands.
 */
processSection
    : 'PROCESS' processCommands
    ;

/**
 * @brief Defines an import section starting with 'IMPORT'.
 *
 * The import section includes a list of import commands following the 'IMPORT' keyword and an end of line.
 */
importSection
    : 'IMPORT' EOL importCommands
    ;

/**
 * @brief Defines a limit section starting with 'LIMIT'.
 *
 * The limit section specifies a numeric limit followed by an end of line.
 */
limitSection
    : 'LIMIT' NUMBER EOL
    ;

/**
 * @brief Contains one or more process commands.
 *
 * Each command consists of one or more EVENT_IDs followed by an end of line.
 */
processCommands
    : (EVENT_ID+ EOL)+
    ;

/**
 * @brief Contains one or more import commands.
 *
 * Each command consists of a number, an EVENT_ID, and an end of line.
 */
importCommands
    : (NUMBER EVENT_ID EOL)+
    ;

/**
 * @brief Represents a gate identifier.
 *
 * A gate identifier is defined as an EVENT_ID.
 */
gateId
    : EVENT_ID
    ;

/**
 * @brief Represents the type of a gate.
 *
 * A gate type can be either '*' or '+'.
 */
gateType
    : '*' | '+'
    ;

/**
 * @brief Represents a child reference in a gate.
 *
 * A child reference is defined as an EVENT_ID.
 */
childRef
    : EVENT_ID
    ;

/**
 * @brief Represents metadata arguments.
 *
 * Metadata arguments can be one of the following: metaEncoding, metaCmd, metaDbName, or metaFTitle.
 */
metaArgs
    : metaEncoding
    | metaCmd
    | metaDbName
    | metaFTitle
    ;

/**
 * @brief Defines the character encoding metadata.
 *
 * The character encoding is specified as '**CHAR32' followed by an end of line.
 */
metaEncoding
    : '**CHAR32' EOL
    ;

/**
 * @brief Defines a command metadata.
 *
 * The command is specified as '*XEQ' followed by an end of line.
 */
metaCmd
    : '*XEQ' EOL
    ;

/**
 * @brief Defines the database name metadata.
 *
 * The database name is specified with '**DBNAME:' followed by any characters until an end of line.
 */
metaDbName
    : '**DBNAME:' .*? EOL
    ;

/**
 * @brief Defines the file title metadata.
 *
 * The file title is specified with '**FTITLE:' followed by any characters until an end of line.
 */
metaFTitle
    : '**FTITLE:' .*? EOL
    ;

/**
 * @brief Defines the format for an event identifier.
 *
 * An EVENT_ID consists of alphanumeric characters, underscores, slashes, or dashes.
 */
EVENT_ID
    : [A-Za-z0-9_/\-]+
    ;

/**
 * @brief Defines the format for a number.
 *
 * A number can be an integer or a floating-point number, optionally preceded by a sign and optionally followed by an exponent.
 */
NUMBER
    : ('+'|'-')?[0-9]+ ('.' [0-9]+)? ([Ee] [+-]? [0-9]+)?
    ;

/**
 * @brief Represents an end of line.
 *
 * An EOL can be a carriage return, a line feed, or both.
 */
EOL
    : [\r\n]+
    ;

/**
 * @brief Represents whitespace.
 *
 * Whitespace includes spaces and tabs and is skipped in the lexical analysis.
 */
WS
    : [ \t]+ -> skip
    ;